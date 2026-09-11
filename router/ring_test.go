package router

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
	"testing"
)

// The two properties the whole design rests on are asserted directly:
// TestLoadNeverExceedsTheBound (bounded loads actually bounds) and
// TestPlainHashingHotSpotsAndBoundedLoadsDoesNot (the reason for the bound).
//
// A ring that quietly stopped honouring the cap would still route every
// request and still hit its prefixes; the only symptom would be one replica at
// 100% while the others idle, which looks like a capacity problem rather than
// a routing bug.

func newTestRing(t *testing.T, n int, epsilon float64) *Ring {
	t.Helper()
	r := NewRing(DefaultVirtualNodes, epsilon)
	for i := 0; i < n; i++ {
		id := fmt.Sprintf("replica-%d", i)
		r.Add(id, fmt.Sprintf("localhost:%d", 9100+i))
		// Add leaves a replica unready on purpose; these tests are about
		// placement, so they stand in for the health checker.
		r.SetReady(id, true)
	}
	return r
}

// ---------------------------------------------------------------------------
// Basics
// ---------------------------------------------------------------------------

func TestEmptyRingReportsNoReplicas(t *testing.T) {
	r := NewRing(DefaultVirtualNodes, DefaultEpsilon)
	if _, err := r.Pick("anything"); err != ErrNoReplicas {
		t.Fatalf("want ErrNoReplicas, got %v", err)
	}
}

func TestSameKeyGoesToSameReplica(t *testing.T) {
	r := newTestRing(t, 8, DefaultEpsilon)

	first, err := r.Pick("shared-system-prompt")
	if err != nil {
		t.Fatal(err)
	}
	// Release so load does not push later picks past the cap; this is testing
	// determinism, not the bound.
	r.Release(first.ID)

	for i := 0; i < 50; i++ {
		got, err := r.Pick("shared-system-prompt")
		if err != nil {
			t.Fatal(err)
		}
		if got.ID != first.ID {
			t.Fatalf("iteration %d: key moved from %s to %s", i, first.ID, got.ID)
		}
		r.Release(got.ID)
	}
}

func TestDifferentKeysSpread(t *testing.T) {
	r := newTestRing(t, 8, DefaultEpsilon)
	seen := map[string]bool{}
	for i := 0; i < 400; i++ {
		rep, err := r.Pick(fmt.Sprintf("key-%d", i))
		if err != nil {
			t.Fatal(err)
		}
		seen[rep.ID] = true
		r.Release(rep.ID)
	}
	if len(seen) < 6 {
		t.Fatalf("400 distinct keys reached only %d of 8 replicas", len(seen))
	}
}

func TestAddingAReplicaMovesFewKeys(t *testing.T) {
	// The consistency half of consistent hashing: adding a node must remap
	// roughly 1/n of keys, not rehash everything. A modulo-based scheme would
	// move almost all of them and invalidate every replica's KV cache at once.
	r := newTestRing(t, 8, math.Inf(1)) // no load bound: isolate placement
	before := map[string]string{}
	for i := 0; i < 2000; i++ {
		k := fmt.Sprintf("key-%d", i)
		rep, _ := r.Pick(k)
		before[k] = rep.ID
		r.Release(rep.ID)
	}

	r.Add("replica-8", "localhost:9108")
	r.SetReady("replica-8", true)

	moved := 0
	for k, was := range before {
		rep, _ := r.Pick(k)
		if rep.ID != was {
			moved++
		}
		r.Release(rep.ID)
	}

	frac := float64(moved) / float64(len(before))
	// Ideal is 1/9 = 11%. Allow generous slack for ring granularity; the point
	// is that it is nowhere near the ~89% a modulo scheme would move.
	if frac > 0.25 {
		t.Fatalf("adding 1 of 9 replicas moved %.1f%% of keys; want ~11%%", frac*100)
	}
}

func TestRemovingAReplicaOnlyMovesItsKeys(t *testing.T) {
	r := newTestRing(t, 6, math.Inf(1))
	before := map[string]string{}
	for i := 0; i < 1500; i++ {
		k := fmt.Sprintf("key-%d", i)
		rep, _ := r.Pick(k)
		before[k] = rep.ID
		r.Release(rep.ID)
	}

	r.Remove("replica-3")

	for k, was := range before {
		rep, _ := r.Pick(k)
		if was != "replica-3" && rep.ID != was {
			t.Fatalf("key %q moved from %s to %s despite its replica surviving",
				k, was, rep.ID)
		}
		r.Release(rep.ID)
	}
}

// ---------------------------------------------------------------------------
// The bound
// ---------------------------------------------------------------------------

func TestLoadNeverExceedsTheBound(t *testing.T) {
	const epsilon = 0.25
	r := newTestRing(t, 8, epsilon)

	// Every request uses the SAME key. Plain consistent hashing would put all
	// 200 on one replica; the bound must spread them.
	for i := 0; i < 200; i++ {
		if _, err := r.Pick("one-very-popular-prefix"); err != nil {
			t.Fatal(err)
		}
	}

	snap := r.Snapshot()
	total := 0
	for _, rep := range snap {
		total += rep.Load
	}
	if total != 200 {
		t.Fatalf("accounted %d of 200 requests", total)
	}

	limit := loadCap(total, len(snap), epsilon)
	for _, rep := range snap {
		if rep.Load > limit {
			t.Errorf("%s holds %d, cap is %d", rep.ID, rep.Load, limit)
		}
	}
}

func TestPlainHashingHotSpotsAndBoundedLoadsDoesNot(t *testing.T) {
	// The reason bounded loads exists, measured against the alternative on a
	// Zipfian prefix distribution -- which is what real traffic looks like.
	const requests = 2000
	zipf := func(rnd *rand.Rand) string {
		// A handful of prefixes dominate, matching a shared system prompt.
		ranks := []int{1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6, 7, 8}
		return fmt.Sprintf("prefix-%d", ranks[rnd.Intn(len(ranks))])
	}

	measure := func(epsilon float64) (maxLoad, minLoad int) {
		r := newTestRing(t, 8, epsilon)
		rnd := rand.New(rand.NewSource(42))
		for i := 0; i < requests; i++ {
			if _, err := r.Pick(zipf(rnd)); err != nil {
				t.Fatal(err)
			}
		}
		snap := r.Snapshot()
		maxLoad, minLoad = 0, math.MaxInt32
		for _, rep := range snap {
			if rep.Load > maxLoad {
				maxLoad = rep.Load
			}
			if rep.Load < minLoad {
				minLoad = rep.Load
			}
		}
		return
	}

	plainMax, plainMin := measure(math.Inf(1)) // no bound
	boundMax, boundMin := measure(0.25)

	t.Logf("plain consistent hashing: max=%d min=%d", plainMax, plainMin)
	t.Logf("bounded loads (eps=0.25): max=%d min=%d", boundMax, boundMin)

	if plainMax <= boundMax {
		t.Fatalf("plain hashing should hot-spot worse: plain max %d, bounded max %d",
			plainMax, boundMax)
	}
	if plainMin != 0 {
		t.Logf("note: plain hashing left its least-loaded replica at %d", plainMin)
	}
	// The bounded ring must be close to even.
	if boundMax > (requests/8)*3/2 {
		t.Errorf("bounded max %d is far above the mean %d", boundMax, requests/8)
	}
	_ = boundMin
}

func TestLoadCapUsesCeilingSoAPlacementAlwaysExists(t *testing.T) {
	// With floor, small totals give a cap of 0 and nothing can be placed.
	if got := loadCap(1, 8, 0.25); got < 1 {
		t.Fatalf("cap for 1 request over 8 replicas is %d; must be at least 1", got)
	}
	// n * cap must always exceed the total, or the pigeonhole argument fails.
	for _, total := range []int{0, 1, 3, 7, 100, 1001} {
		for _, n := range []int{1, 3, 8, 64} {
			cap := loadCap(total, n, 0.25)
			if n*cap <= total && total > 0 {
				t.Errorf("total=%d n=%d cap=%d: no valid assignment exists",
					total, n, cap)
			}
		}
	}
}

func TestEpsilonTradesEvennessAgainstLocality(t *testing.T) {
	// Smaller epsilon is more even. This is the knob's documented behaviour,
	// so it is asserted rather than assumed.
	spread := func(epsilon float64) int {
		r := newTestRing(t, 8, epsilon)
		for i := 0; i < 800; i++ {
			if _, err := r.Pick("single-hot-prefix"); err != nil {
				t.Fatal(err)
			}
		}
		maxL, minL := 0, math.MaxInt32
		for _, rep := range r.Snapshot() {
			if rep.Load > maxL {
				maxL = rep.Load
			}
			if rep.Load < minL {
				minL = rep.Load
			}
		}
		return maxL - minL
	}
	tight, loose := spread(0.05), spread(2.0)
	if tight > loose {
		t.Errorf("epsilon 0.05 spread %d should not exceed epsilon 2.0 spread %d",
			tight, loose)
	}
}

// ---------------------------------------------------------------------------
// Readiness and lifecycle
// ---------------------------------------------------------------------------

func TestUnreadyReplicasAreNeverPicked(t *testing.T) {
	r := newTestRing(t, 4, DefaultEpsilon)
	r.SetReady("replica-0", false)
	r.SetReady("replica-2", false)

	for i := 0; i < 300; i++ {
		rep, err := r.Pick(fmt.Sprintf("key-%d", i))
		if err != nil {
			t.Fatal(err)
		}
		if rep.ID == "replica-0" || rep.ID == "replica-2" {
			t.Fatalf("picked unready replica %s", rep.ID)
		}
		r.Release(rep.ID)
	}
}

func TestAllUnreadyIsAnError(t *testing.T) {
	r := newTestRing(t, 3, DefaultEpsilon)
	for i := 0; i < 3; i++ {
		r.SetReady(fmt.Sprintf("replica-%d", i), false)
	}
	if _, err := r.Pick("k"); err != ErrNoReplicas {
		t.Fatalf("want ErrNoReplicas when every replica is unready, got %v", err)
	}
}

func TestUnreadyReplicaKeepsItsRingPositions(t *testing.T) {
	// Marking a replica unready must not shift other keys. Dropping its points
	// would remap everything hashing near it and discard cluster-wide locality
	// for what is usually a transient condition.
	r := newTestRing(t, 6, math.Inf(1))
	before := map[string]string{}
	for i := 0; i < 600; i++ {
		k := fmt.Sprintf("key-%d", i)
		rep, _ := r.Pick(k)
		before[k] = rep.ID
		r.Release(rep.ID)
	}

	r.SetReady("replica-4", false)

	for k, was := range before {
		if was == "replica-4" {
			continue // these must move; they have nowhere else to go
		}
		rep, _ := r.Pick(k)
		if rep.ID != was {
			t.Fatalf("key %q moved from %s to %s when an unrelated replica went unready",
				k, was, rep.ID)
		}
		r.Release(rep.ID)
	}
}

func TestReleaseDoesNotUnderflow(t *testing.T) {
	r := newTestRing(t, 2, DefaultEpsilon)
	r.Release("replica-0")
	r.Release("replica-0")
	for _, rep := range r.Snapshot() {
		if rep.Load < 0 {
			t.Fatalf("%s load went negative: %d", rep.ID, rep.Load)
		}
	}
}

func TestRemovingAnUnknownReplicaIsSafe(t *testing.T) {
	r := newTestRing(t, 3, DefaultEpsilon)
	r.Remove("never-existed")
	if r.Len() != 3 {
		t.Fatalf("ring size changed to %d", r.Len())
	}
}

func TestAddIsIdempotent(t *testing.T) {
	r := newTestRing(t, 2, DefaultEpsilon)
	r.Add("replica-0", "localhost:9999")
	if r.Len() != 2 {
		t.Fatalf("re-adding created a duplicate: %d replicas", r.Len())
	}
	for _, rep := range r.Snapshot() {
		if rep.ID == "replica-0" && rep.Addr != "localhost:9999" {
			t.Fatalf("address not updated: %s", rep.Addr)
		}
	}
}

// ---------------------------------------------------------------------------
// Concurrency
// ---------------------------------------------------------------------------

func TestConcurrentPickAndReleaseKeepsAccountingExact(t *testing.T) {
	// The router picks from many goroutines at once. A lost update here shows
	// up as load drifting away from reality, which silently breaks the bound.
	r := newTestRing(t, 8, DefaultEpsilon)

	const workers, each = 16, 200
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for i := 0; i < each; i++ {
				rep, err := r.Pick(fmt.Sprintf("key-%d", i%37))
				if err != nil {
					t.Error(err)
					return
				}
				r.Release(rep.ID)
			}
		}(w)
	}
	wg.Wait()

	for _, rep := range r.Snapshot() {
		if rep.Load != 0 {
			t.Errorf("%s left with load %d after every request was released",
				rep.ID, rep.Load)
		}
	}
}

func TestConcurrentTopologyChangesDoNotCorruptTheRing(t *testing.T) {
	r := newTestRing(t, 4, DefaultEpsilon)
	var wg sync.WaitGroup

	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < 200; i++ {
			id := fmt.Sprintf("churn-%d", i%5)
			r.Add(id, "localhost:9999")
			r.SetReady(id, true)
			r.Remove(id)
		}
	}()

	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < 2000; i++ {
			if rep, err := r.Pick(fmt.Sprintf("key-%d", i)); err == nil {
				r.Release(rep.ID)
			}
		}
	}()

	wg.Wait()
	if r.Len() != 4 {
		t.Fatalf("ring left with %d replicas after churn, want 4", r.Len())
	}
}

// ---------------------------------------------------------------------------
// Prefix keys
// ---------------------------------------------------------------------------

func TestPrefixKeyGroupsSharedSystemPrompts(t *testing.T) {
	system := "You are a helpful assistant. Answer concisely and accurately."
	a := PrefixKey(system+" What is the capital of France?", 48)
	b := PrefixKey(system+" Explain photosynthesis briefly.", 48)
	if a != b {
		t.Fatalf("shared system prompt produced different keys:\n  %q\n  %q", a, b)
	}
}

func TestPrefixKeySeparatesDifferentPrompts(t *testing.T) {
	a := PrefixKey("Translate the following text into French:", 32)
	b := PrefixKey("Summarise the following article:", 32)
	if a == b {
		t.Fatal("different prompts collapsed to the same key")
	}
}

func TestPrefixKeyHandlesShortPrompts(t *testing.T) {
	if got := PrefixKey("hi", 128); got != "hi" {
		t.Fatalf("short prompt was truncated: %q", got)
	}
	if got := PrefixKey("hello", 0); got != "hello" {
		t.Fatalf("zero prefixLen should mean the whole prompt, got %q", got)
	}
}

func TestNewReplicasStartUnready(t *testing.T) {
	// A registry entry says a replica exists, not that it can serve. Marking
	// it ready on Add sends traffic to a process still loading its model --
	// which is exactly what happened the first time three replicas were
	// started: /healthz reported ready instantly and every request failed.
	r := NewRing(DefaultVirtualNodes, DefaultEpsilon)
	r.Add("fresh", "localhost:9101")

	if r.ReadyCount() != 0 {
		t.Fatalf("a newly added replica reported ready: %d", r.ReadyCount())
	}
	if _, err := r.Pick("k"); err != ErrNoReplicas {
		t.Fatalf("picked an unverified replica: %v", err)
	}

	r.SetReady("fresh", true)
	if _, err := r.Pick("k"); err != nil {
		t.Fatalf("still unpickable after health check: %v", err)
	}
}
