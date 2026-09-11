// Package router implements prefix-aware request routing over a replica set.
//
// Consistent hashing with bounded loads. Two mechanisms, and both halves
// matter:
//
// Plain consistent hashing gives cache locality -- the same prefix lands on
// the same replica, so that replica already holds its KV blocks and skips the
// prefill. It also hot-spots badly. Under the Zipfian prefix distribution real
// traffic has, one popular system prompt pins one replica at 100% while the
// rest idle, and consistent hashing has no mechanism to notice.
//
// Bounded loads caps the damage. Each replica may hold at most
//
//	cap = ceil(c * totalLoad / numReplicas),  c = 1 + epsilon
//
// and the walk skips any replica already at the cap. Most requests still land
// on their hashed replica, so locality survives; the overloaded ones spill to
// the next replica clockwise, which is deterministic and therefore still
// cache-friendly for the spill itself.
//
// The ceiling is load-bearing rather than cosmetic. With floor, a small total
// load gives cap 0 and nothing can be placed anywhere. With ceiling, a valid
// assignment provably always exists: n replicas times ceil(c*m/n) >= c*m > m
// for c > 1, so the pigeonhole argument never fails and the walk always
// terminates with a placement.
//
// Reference: Mirrokni, Thorup, Zadimoghaddam, "Consistent Hashing with
// Bounded Loads" (SODA 2018), arXiv:1608.01350.
package router

import (
	"errors"
	"fmt"
	"hash/fnv"
	"math"
	"sort"
	"sync"
)

// DefaultVirtualNodes is how many points each replica occupies on the ring.
//
// Too few and the ring is lumpy: with one point per replica the arc a replica
// owns is a uniform random slice, and the variance in slice size is as large
// as the slices themselves. 150 is the conventional figure -- it brings the
// spread of owned key-space to a few percent while keeping the sorted ring
// small enough to binary-search cheaply.
const DefaultVirtualNodes = 150

// DefaultEpsilon sets the load cap at 1.25x the cluster mean.
//
// Smaller is more even and less local; larger is more local and more skewed.
// At 0 this degenerates to least-loaded routing and throws away every prefix
// hit; at infinity it is plain consistent hashing and hot-spots.
const DefaultEpsilon = 0.25

// ErrNoReplicas is returned when the ring is empty.
var ErrNoReplicas = errors.New("router: no replicas available")

// Replica is one backend, as the ring sees it.
type Replica struct {
	ID   string
	Addr string
	// Load is requests currently in flight on this replica. Updated by the
	// router as it dispatches and completes, not polled -- a health-poll
	// interval of even 100ms is far longer than a routing decision, so polled
	// load would route on state that is already wrong.
	Load int
	// Ready is false while a replica is starting up or draining. An unready
	// replica stays on the ring (so its hash positions do not move and
	// invalidate everyone else's locality) but is never selected.
	Ready bool
}

type ringPoint struct {
	hash     uint64
	replicaID string
}

// Ring is a consistent hash ring with bounded loads. Safe for concurrent use.
type Ring struct {
	mu           sync.RWMutex
	virtualNodes int
	epsilon      float64

	replicas map[string]*Replica
	// points is kept sorted by hash so lookup is a binary search.
	points []ringPoint
}

// NewRing builds an empty ring.
func NewRing(virtualNodes int, epsilon float64) *Ring {
	if virtualNodes <= 0 {
		virtualNodes = DefaultVirtualNodes
	}
	if epsilon < 0 {
		epsilon = DefaultEpsilon
	}
	return &Ring{
		virtualNodes: virtualNodes,
		epsilon:      epsilon,
		replicas:     make(map[string]*Replica),
	}
}

// Add inserts a replica, or updates its address if already present.
func (r *Ring) Add(id, addr string) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if existing, ok := r.replicas[id]; ok {
		existing.Addr = addr
		return
	}
	// Unready until a health check says otherwise. Trusting a replica the
	// moment it is registered means routing to a process that is still loading
	// its model -- the registry says it exists, not that it can serve. The
	// health checker promotes it on the first successful poll.
	r.replicas[id] = &Replica{ID: id, Addr: addr, Ready: false}
	for i := 0; i < r.virtualNodes; i++ {
		r.points = append(r.points, ringPoint{
			hash:      hashKey(fmt.Sprintf("%s#%d", id, i)),
			replicaID: id,
		})
	}
	r.sortPoints()
}

// Remove drops a replica and all of its ring points.
func (r *Ring) Remove(id string) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if _, ok := r.replicas[id]; !ok {
		return
	}
	delete(r.replicas, id)

	kept := r.points[:0]
	for _, p := range r.points {
		if p.replicaID != id {
			kept = append(kept, p)
		}
	}
	r.points = kept
}

// SetReady marks a replica selectable or not.
//
// An unready replica keeps its ring positions. Removing them would shift every
// key that hashes near it onto a different replica, discarding cache locality
// cluster-wide for what is usually a transient condition.
func (r *Ring) SetReady(id string, ready bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if rep, ok := r.replicas[id]; ok {
		rep.Ready = ready
	}
}

// Pick returns the replica that should serve key, honouring the load bound.
//
// The key is a prefix of the request -- see PrefixKey. Two requests sharing a
// system prompt produce the same key and therefore prefer the same replica,
// which is what makes its KV cache worth anything.
func (r *Ring) Pick(key string) (*Replica, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if len(r.points) == 0 {
		return nil, ErrNoReplicas
	}

	ready, total := 0, 0
	for _, rep := range r.replicas {
		if rep.Ready {
			ready++
			total += rep.Load
		}
	}
	if ready == 0 {
		return nil, ErrNoReplicas
	}

	limit := loadCap(total+1, ready, r.epsilon)

	// Walk clockwise from the key's position. Each distinct replica is
	// considered once; the first ready one under the cap wins.
	start := r.search(hashKey(key))
	seen := make(map[string]bool, ready)

	for i := 0; i < len(r.points) && len(seen) < ready; i++ {
		p := r.points[(start+i)%len(r.points)]
		if seen[p.replicaID] {
			continue
		}
		rep := r.replicas[p.replicaID]
		if rep == nil || !rep.Ready {
			continue
		}
		seen[p.replicaID] = true
		if rep.Load < limit {
			rep.Load++
			return rep, nil
		}
	}

	// Unreachable given the ceiling in loadCap -- n*ceil(c*m/n) > m for c > 1,
	// so some replica is always under the cap. Falling back to least-loaded
	// rather than erroring, because dropping a request to preserve a proof is
	// the wrong trade in a server.
	var best *Replica
	for _, rep := range r.replicas {
		if rep.Ready && (best == nil || rep.Load < best.Load) {
			best = rep
		}
	}
	if best == nil {
		return nil, ErrNoReplicas
	}
	best.Load++
	return best, nil
}

// Release records that a request finished on a replica.
func (r *Ring) Release(id string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if rep, ok := r.replicas[id]; ok && rep.Load > 0 {
		rep.Load--
	}
}

// Snapshot returns a copy of current replica state, for metrics and tests.
func (r *Ring) Snapshot() []Replica {
	r.mu.RLock()
	defer r.mu.RUnlock()

	out := make([]Replica, 0, len(r.replicas))
	for _, rep := range r.replicas {
		out = append(out, *rep)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out
}

// Len is the number of replicas on the ring, ready or not.
func (r *Ring) Len() int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return len(r.replicas)
}

// ReadyCount is the number of selectable replicas.
func (r *Ring) ReadyCount() int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	n := 0
	for _, rep := range r.replicas {
		if rep.Ready {
			n++
		}
	}
	return n
}

// LoadCap exposes the current per-replica bound, for metrics.
func (r *Ring) LoadCap() int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	ready, total := 0, 0
	for _, rep := range r.replicas {
		if rep.Ready {
			ready++
			total += rep.Load
		}
	}
	if ready == 0 {
		return 0
	}
	return loadCap(total, ready, r.epsilon)
}

// loadCap is ceil((1+epsilon) * total / n), with a floor of 1.
//
// The ceiling guarantees a placement exists. The floor of 1 covers the empty
// cluster: with total 0 the cap would be 0 and the very first request would
// find every replica "at capacity".
func loadCap(total, n int, epsilon float64) int {
	if n <= 0 {
		return 0
	}
	// A non-finite epsilon means "no bound". Computing with it produces Inf or
	// NaN, and int64(NaN) is undefined in Go -- which silently yielded a cap of
	// 1, the tightest possible bound, when the caller asked for the loosest.
	if math.IsInf(epsilon, 1) || math.IsNaN(epsilon) {
		return math.MaxInt32
	}
	c := 1.0 + epsilon
	limit := int(ceilDiv(c*float64(total), float64(n)))
	if limit < 1 {
		return 1
	}
	return limit
}

func ceilDiv(a, b float64) float64 {
	q := a / b
	if q == float64(int64(q)) {
		return q
	}
	return float64(int64(q) + 1)
}

// search returns the index of the first ring point at or after h.
func (r *Ring) search(h uint64) int {
	i := sort.Search(len(r.points), func(i int) bool {
		return r.points[i].hash >= h
	})
	if i == len(r.points) {
		return 0 // wrap around
	}
	return i
}

func (r *Ring) sortPoints() {
	sort.Slice(r.points, func(i, j int) bool {
		return r.points[i].hash < r.points[j].hash
	})
}

// hashKey maps a string onto the ring.
//
// FNV-1a for speed, then a 64-bit avalanche finalizer -- and the finalizer is
// not optional. FNV-1a alone has weak avalanche on short, similar inputs, and
// every input here is short and similar: keys are prompt prefixes that share a
// system prompt, and ring points are "replica-0#0", "replica-0#1", ... Raw
// FNV-1a put those into narrow bands, and 390 of 400 distinct keys landed on a
// single replica. The ring looked fine -- 1,200 points, correct lookups -- and
// routed almost everything to one backend.
//
// fmix64 is MurmurHash3's finalizer: two xor-shift-multiply rounds that spread
// any single-bit input change across all 64 output bits. Cheap, and it is what
// makes uniform ring placement true rather than assumed. A cryptographic hash
// would also work and would cost far more on a per-request path.
func hashKey(s string) uint64 {
	h := fnv.New64a()
	_, _ = h.Write([]byte(s))
	return fmix64(h.Sum64())
}

func fmix64(k uint64) uint64 {
	k ^= k >> 33
	k *= 0xff51afd7ed558ccd
	k ^= k >> 33
	k *= 0xc4ceb9fe1a85ec53
	k ^= k >> 33
	return k
}

// PrefixKey is the routing key for a prompt: its first prefixLen bytes.
//
// Bytes rather than tokens, deliberately. Tokenising in the router would mean
// shipping the tokenizer to the control plane and paying for it on every
// request, and it buys nothing: two prompts sharing a system prompt share a
// byte prefix exactly when they share a token prefix, because tokenization is
// deterministic and left-to-right.
func PrefixKey(prompt string, prefixLen int) string {
	if prefixLen <= 0 || len(prompt) <= prefixLen {
		return prompt
	}
	return prompt[:prefixLen]
}
