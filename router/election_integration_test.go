package router

import (
	"context"
	"sync"
	"testing"
	"time"
)

// TestElectionGrantsExactlyOneLeader is the property the autoscaler depends
// on. Two controllers observing the same overloaded cluster each compute the
// same delta; if both apply it the cluster overshoots by 2x, then both see the
// overshoot and scale down, and the pair oscillates forever.
func TestElectionGrantsExactlyOneLeader(t *testing.T) {
	endpoints := etcdEndpoints(t)
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	var (
		mu       sync.Mutex
		active   int
		maxSeen  int
		everHeld = map[string]bool{}
	)

	enter := func(id string) {
		mu.Lock()
		defer mu.Unlock()
		active++
		everHeld[id] = true
		if active > maxSeen {
			maxSeen = active
		}
	}
	leave := func() {
		mu.Lock()
		defer mu.Unlock()
		active--
	}

	const candidates = 3
	var wg sync.WaitGroup
	for i := 0; i < candidates; i++ {
		id := string(rune('a' + i))
		wg.Add(1)
		go func() {
			defer wg.Done()
			_ = RunAsLeader(ctx, endpoints, id, 5, quietLogger(),
				func(leaderCtx context.Context) {
					enter(id)
					defer leave()
					// Hold leadership briefly, then step down so the next
					// candidate gets a turn and the test observes a handover
					// rather than only a first election.
					select {
					case <-time.After(2 * time.Second):
					case <-leaderCtx.Done():
					}
				})
		}()
	}

	// Let leadership change hands a few times.
	time.Sleep(8 * time.Second)
	cancel()

	done := make(chan struct{})
	go func() { wg.Wait(); close(done) }()
	select {
	case <-done:
	case <-time.After(20 * time.Second):
		t.Fatal("candidates did not shut down after context cancellation")
	}

	mu.Lock()
	defer mu.Unlock()
	if maxSeen > 1 {
		t.Fatalf("%d controllers held leadership simultaneously; the whole "+
			"point of the election is that scaling actions are serialised",
			maxSeen)
	}
	if maxSeen == 0 {
		t.Fatal("nobody was ever elected; an autoscaler that never runs is " +
			"indistinguishable from a cluster that never needs scaling, " +
			"which is why this is worth asserting")
	}
	if len(everHeld) < 2 {
		t.Fatalf("only %d candidate(s) ever held leadership; a resigning "+
			"leader must hand over rather than reacquire, or failover is "+
			"untested", len(everHeld))
	}
	t.Logf("leadership passed through %d of %d candidates, never overlapping",
		len(everHeld), candidates)
}

// TestLeaderContextEndsOnResign checks the half that is easy to leave out.
// Winning an election is not the hard part; noticing you have lost it is.
// A leader that keeps acting after its lease lapsed is exactly the split-brain
// the election was supposed to prevent, and harder to spot because the logs
// still show one leader per instance.
func TestLeaderContextEndsOnResign(t *testing.T) {
	endpoints := etcdEndpoints(t)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	observed := make(chan context.Context, 1)
	go func() {
		_ = RunAsLeader(ctx, endpoints, "solo", 5, quietLogger(),
			func(leaderCtx context.Context) {
				select {
				case observed <- leaderCtx:
				default:
				}
				<-leaderCtx.Done()
			})
	}()

	var leaderCtx context.Context
	select {
	case leaderCtx = <-observed:
	case <-time.After(15 * time.Second):
		t.Fatal("never elected")
	}

	if leaderCtx.Err() != nil {
		t.Fatal("leader context was already cancelled on entry")
	}
	cancel()

	select {
	case <-leaderCtx.Done():
	case <-time.After(10 * time.Second):
		t.Fatal("leader context outlived the parent; a leader that does not " +
			"notice it has stopped being the leader keeps scaling alongside " +
			"its replacement")
	}
}
