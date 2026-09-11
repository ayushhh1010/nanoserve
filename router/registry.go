package router

import (
	"context"
	"log/slog"
	"sync"
	"time"
)

// Registry is where the router learns which replicas exist.
//
// An interface with two implementations on purpose. The static one makes the
// router runnable and fully testable with no infrastructure; the etcd one is
// what a real deployment uses. The router itself never learns which it has,
// so a chaos test can kill etcd and assert the router keeps serving from the
// endpoints it already knows -- which is the actual requirement, not an
// implementation detail.
type Registry interface {
	// Watch streams membership changes until ctx is cancelled. The first
	// event carries the full current set, so a caller never has to fetch and
	// then subscribe and reconcile the gap between them.
	Watch(ctx context.Context) (<-chan []Endpoint, error)
	// Close releases whatever the implementation holds.
	Close() error
}

// Endpoint is one registered replica.
type Endpoint struct {
	ID   string
	Addr string
}

// StaticRegistry serves a fixed set, supplied at construction.
//
// Not a stub: running with a known replica list is a legitimate deployment
// (a fixed Docker Compose, a pinned StatefulSet), and it is the configuration
// every test uses.
type StaticRegistry struct {
	endpoints []Endpoint
}

func NewStaticRegistry(endpoints []Endpoint) *StaticRegistry {
	return &StaticRegistry{endpoints: endpoints}
}

func (s *StaticRegistry) Watch(ctx context.Context) (<-chan []Endpoint, error) {
	ch := make(chan []Endpoint, 1)
	ch <- append([]Endpoint(nil), s.endpoints...)
	go func() {
		<-ctx.Done()
		close(ch)
	}()
	return ch, nil
}

func (s *StaticRegistry) Close() error { return nil }

// HealthChecker keeps ring readiness and load in step with reality.
//
// Membership says a replica exists; health says whether it can take work.
// They are separate because they fail separately: a replica can be registered
// and overloaded, or unregistered and still finishing in-flight requests.
type HealthChecker struct {
	ring     *Ring
	pool     *ClientPool
	interval time.Duration
	timeout  time.Duration
	log      *slog.Logger

	mu     sync.Mutex
	misses map[string]int
	// last successful health response per replica. The checker already pays
	// for these RPCs to decide readiness; the load fields ride along for free
	// and are what the autoscaler steers on. Polling them a second time from
	// a separate component would double the health traffic and give the two
	// consumers disagreeing views of the same instant.
	loads map[string]*HealthResponse
	// failuresToUnready is how many consecutive failed checks mark a replica
	// unready. One is too eager: a single dropped health check during a long
	// forward pass would pull a healthy replica out of rotation and move every
	// key it owned.
	failuresToUnready int
}

func NewHealthChecker(
	ring *Ring, pool *ClientPool, interval, timeout time.Duration, log *slog.Logger,
) *HealthChecker {
	return &HealthChecker{
		ring:              ring,
		pool:              pool,
		interval:          interval,
		timeout:           timeout,
		log:               log,
		misses:            make(map[string]int),
		loads:             make(map[string]*HealthResponse),
		failuresToUnready: 2,
	}
}

// Run polls every replica until ctx is cancelled.
func (h *HealthChecker) Run(ctx context.Context) {
	ticker := time.NewTicker(h.interval)
	defer ticker.Stop()

	h.checkAll(ctx)
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			h.checkAll(ctx)
		}
	}
}

func (h *HealthChecker) checkAll(ctx context.Context) {
	replicas := h.ring.Snapshot()
	var wg sync.WaitGroup
	// Concurrently: a serial sweep takes len(replicas) * timeout in the worst
	// case, which at any real replica count is longer than the interval, so
	// the checker would fall permanently behind exactly when replicas are
	// failing.
	for _, rep := range replicas {
		wg.Add(1)
		go func(id string) {
			defer wg.Done()
			h.checkOne(ctx, id)
		}(rep.ID)
	}
	wg.Wait()
}

func (h *HealthChecker) checkOne(ctx context.Context, id string) {
	cctx, cancel := context.WithTimeout(ctx, h.timeout)
	defer cancel()

	resp, err := h.pool.Health(cctx, id)
	h.mu.Lock()
	defer h.mu.Unlock()

	if err != nil || !resp.GetReady() {
		h.misses[id]++
		delete(h.loads, id)
		if h.misses[id] >= h.failuresToUnready {
			h.ring.SetReady(id, false)
			if h.misses[id] == h.failuresToUnready {
				reason := "not ready"
				if err != nil {
					reason = err.Error()
				}
				h.log.Warn("replica unready", "replica", id, "reason", reason)
			}
		}
		return
	}

	if h.misses[id] > 0 {
		h.log.Info("replica recovered", "replica", id)
	}
	h.misses[id] = 0
	h.loads[id] = resp
	h.ring.SetReady(id, true)
}

// Load returns the fleet's current aggregate, as the autoscaler sees it.
//
// Only replicas that are ready *and* answered their last check are counted. A
// replica that has stopped answering contributes no queue depth, which is
// correct: its requests are already being retried onto replicas that do count
// them, and counting a dead replica as capacity would suppress the scale-up
// that its death is precisely the reason for.
func (h *HealthChecker) Load() ClusterLoad {
	h.mu.Lock()
	defer h.mu.Unlock()

	var out ClusterLoad
	for _, rep := range h.ring.Snapshot() {
		out.Registered++
		if !rep.Ready {
			continue
		}
		resp, ok := h.loads[rep.ID]
		if !ok || h.misses[rep.ID] > 0 {
			continue
		}
		out.ReadyReplicas++
		out.TotalWaiting += int(resp.GetNumWaiting())
		out.TotalRunning += int(resp.GetNumRunning())
		if kv := float64(resp.GetKvUtilization()); kv > out.MaxKVUtilization {
			out.MaxKVUtilization = kv
		}
	}
	return out
}

// Reconciler applies registry membership changes to the ring.
type Reconciler struct {
	ring *Ring
	log  *slog.Logger
}

func NewReconciler(ring *Ring, log *slog.Logger) *Reconciler {
	return &Reconciler{ring: ring, log: log}
}

// Run consumes membership events until the channel closes or ctx is cancelled.
func (rc *Reconciler) Run(ctx context.Context, events <-chan []Endpoint) {
	for {
		select {
		case <-ctx.Done():
			return
		case eps, ok := <-events:
			if !ok {
				return
			}
			rc.apply(eps)
		}
	}
}

func (rc *Reconciler) apply(endpoints []Endpoint) {
	wanted := make(map[string]string, len(endpoints))
	for _, e := range endpoints {
		wanted[e.ID] = e.Addr
	}

	for _, rep := range rc.ring.Snapshot() {
		if _, keep := wanted[rep.ID]; !keep {
			rc.log.Info("replica deregistered", "replica", rep.ID)
			rc.ring.Remove(rep.ID)
		}
	}
	for id, addr := range wanted {
		rc.ring.Add(id, addr)
	}
}
