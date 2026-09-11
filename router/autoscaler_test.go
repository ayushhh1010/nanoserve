package router

import (
	"context"
	"os/exec"
	"runtime"
	"testing"
	"time"
)

func testPolicy() ScalePolicy {
	p := DefaultScalePolicy()
	p.MinReplicas = 1
	p.MaxReplicas = 8
	p.TargetQueueDepth = 4
	return p
}

// newTestAutoscaler gives the controller a clock the test drives, so the
// five-minute stabilization window can be exercised without a test that takes
// five minutes -- and, more importantly, without a test whose result depends
// on how loaded the CI machine was.
func newTestAutoscaler(t *testing.T, p ScalePolicy) (*Autoscaler, *fakeClock) {
	t.Helper()
	a, err := NewAutoscaler(p, quietLogger())
	if err != nil {
		t.Fatalf("NewAutoscaler: %v", err)
	}
	clk := &fakeClock{t: time.Unix(1_700_000_000, 0)}
	a.now = clk.Now
	return a, clk
}

type fakeClock struct{ t time.Time }

func (c *fakeClock) Now() time.Time          { return c.t }
func (c *fakeClock) Advance(d time.Duration) { c.t = c.t.Add(d) }

func TestScalePolicyValidation(t *testing.T) {
	cases := map[string]func(*ScalePolicy){
		"zero target":       func(p *ScalePolicy) { p.TargetQueueDepth = 0 },
		"negative target":   func(p *ScalePolicy) { p.TargetQueueDepth = -1 },
		"min below one":     func(p *ScalePolicy) { p.MinReplicas = 0 },
		"max below min":     func(p *ScalePolicy) { p.MinReplicas, p.MaxReplicas = 5, 4 },
		"tolerance at one":  func(p *ScalePolicy) { p.Tolerance = 1 },
		"kv high water two": func(p *ScalePolicy) { p.KVHighWater = 2 },
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			p := testPolicy()
			mutate(&p)
			if _, err := NewAutoscaler(p, quietLogger()); err == nil {
				t.Fatal("expected a misconfigured policy to be rejected at " +
					"construction; a bad policy discovered at runtime scales " +
					"a production cluster before anyone reads the log")
			}
		})
	}
}

func TestScaleUpIsProportional(t *testing.T) {
	a, _ := newTestAutoscaler(t, testPolicy())

	// 4 replicas, 32 waiting => 8/replica against a target of 4 => ratio 2.
	d := a.Decide(ClusterLoad{Registered: 4, ReadyReplicas: 4, TotalWaiting: 32})
	if d.Desired != 8 {
		t.Fatalf("desired = %d, want 8 (ceil(4 * 8/4)); reason: %s", d.Desired, d.Reason)
	}
}

func TestScaleUpClampsToMax(t *testing.T) {
	p := testPolicy()
	p.MaxReplicas = 6
	a, _ := newTestAutoscaler(t, p)

	d := a.Decide(ClusterLoad{Registered: 4, ReadyReplicas: 4, TotalWaiting: 400})
	if d.Desired != 6 {
		t.Fatalf("desired = %d, want the ceiling 6", d.Desired)
	}
}

func TestToleranceSuppressesNoise(t *testing.T) {
	a, _ := newTestAutoscaler(t, testPolicy())

	// 4 replicas, 17 waiting => 4.25/replica, ratio 1.0625, inside the 10%
	// band. Acting here is what makes a controller oscillate: the action
	// changes the metric, which triggers the next action.
	d := a.Decide(ClusterLoad{Registered: 4, ReadyReplicas: 4, TotalWaiting: 17})
	if d.Changed() {
		t.Fatalf("scaled to %d on a %.1f%% deviation; the dead band is what "+
			"stops the controller chasing its own tail (reason: %s)",
			d.Desired, 6.25, d.Reason)
	}
}

func TestScaleDownRequiresSustainedEvidence(t *testing.T) {
	p := testPolicy()
	p.ScaleDownStabilization = 5 * time.Minute
	a, clk := newTestAutoscaler(t, p)

	busy := ClusterLoad{Registered: 4, ReadyReplicas: 4, TotalWaiting: 16}
	quiet := ClusterLoad{Registered: 4, ReadyReplicas: 4, TotalWaiting: 0}

	if d := a.Decide(busy); d.Changed() {
		t.Fatalf("busy cluster at exactly target scaled to %d", d.Desired)
	}

	// Traffic stops. The first quiet observation must NOT give up a replica:
	// one idle moment inside a busy window is a gap between bursts, and the
	// replica being discarded holds a warm prefix cache.
	clk.Advance(30 * time.Second)
	d := a.Decide(quiet)
	if d.Changed() {
		t.Fatalf("scaled down to %d after 30s of quiet; the %v window exists "+
			"precisely to refuse this", d.Desired, p.ScaleDownStabilization)
	}
	if !d.Held {
		t.Fatal("a suppressed scale-down must report Held, so an operator can " +
			"tell 'nothing to do' from 'waiting for evidence'")
	}

	// Sustained quiet past the window: now it is real.
	for i := 0; i < 12; i++ {
		clk.Advance(30 * time.Second)
		d = a.Decide(quiet)
	}
	if d.Desired != p.MinReplicas {
		t.Fatalf("desired = %d after %v of sustained quiet, want the floor %d "+
			"(reason: %s)", d.Desired, 6*time.Minute, p.MinReplicas, d.Reason)
	}
}

// TestBurstInsideWindowCancelsScaleDown is the property the window is for.
func TestBurstInsideWindowCancelsScaleDown(t *testing.T) {
	p := testPolicy()
	p.ScaleDownStabilization = 5 * time.Minute
	a, clk := newTestAutoscaler(t, p)

	quiet := ClusterLoad{Registered: 4, ReadyReplicas: 4, TotalWaiting: 0}
	burst := ClusterLoad{Registered: 4, ReadyReplicas: 4, TotalWaiting: 32}

	for i := 0; i < 8; i++ {
		clk.Advance(30 * time.Second)
		a.Decide(quiet)
	}
	// A burst arrives just before the window would have elapsed.
	clk.Advance(30 * time.Second)
	if d := a.Decide(burst); d.Desired != 8 {
		t.Fatalf("burst scaled to %d, want 8: scale-up must be immediate", d.Desired)
	}
	// Quiet again. The burst is still inside the window, so the highest
	// recent demand must keep the replicas alive.
	clk.Advance(30 * time.Second)
	d := a.Decide(quiet)
	if d.Desired < 4 {
		t.Fatalf("scaled to %d immediately after a burst still inside the %v "+
			"window; the window must remember the peak (reason: %s)",
			d.Desired, p.ScaleDownStabilization, d.Reason)
	}
}

func TestKVPressureForcesScaleUpWithEmptyQueue(t *testing.T) {
	a, _ := newTestAutoscaler(t, testPolicy())

	// Empty queue, but KV nearly exhausted: a handful of very long sequences.
	// Queue depth alone says "shrink"; KV says the next long prompt starts
	// preempting. The guard must win.
	d := a.Decide(ClusterLoad{
		Registered: 2, ReadyReplicas: 2, TotalWaiting: 0,
		TotalRunning: 2, MaxKVUtilization: 0.97,
	})
	if d.Desired <= 2 {
		t.Fatalf("desired = %d with KV at 97%% and an empty queue; KV is the "+
			"real capacity limit and does not appear in queue depth until "+
			"preemption has already started (reason: %s)", d.Desired, d.Reason)
	}
}

func TestKVGuardNeverForcesScaleDown(t *testing.T) {
	a, _ := newTestAutoscaler(t, testPolicy())

	// High KV *and* a long queue: the proportional result already exceeds the
	// guard, and the guard must not pull it back down.
	d := a.Decide(ClusterLoad{
		Registered: 2, ReadyReplicas: 2, TotalWaiting: 40, MaxKVUtilization: 0.95,
	})
	if d.Desired != 8 {
		t.Fatalf("desired = %d, want 8: the KV guard raises a floor, it does "+
			"not cap the proportional result (reason: %s)", d.Desired, d.Reason)
	}
}

// TestEmptyRegistryRequestsFloor and TestUnhealthyFleetHolds are the two
// zeros. They look identical in a naive metric and want opposite responses.
func TestEmptyRegistryRequestsFloor(t *testing.T) {
	a, _ := newTestAutoscaler(t, testPolicy())
	d := a.Decide(ClusterLoad{Registered: 0, ReadyReplicas: 0})
	if d.Desired != 1 {
		t.Fatalf("desired = %d with an empty registry, want the floor 1", d.Desired)
	}
}

func TestUnhealthyFleetHoldsRatherThanMultiplies(t *testing.T) {
	a, _ := newTestAutoscaler(t, testPolicy())

	// Four replicas registered, none healthy, and a queue backing up behind
	// them. The proportional formula is undefined here (divide by zero ready);
	// worse, treating it as a capacity shortage would launch replicas into a
	// fleet that is failing for its own reasons.
	d := a.Decide(ClusterLoad{Registered: 4, ReadyReplicas: 0, TotalWaiting: 500})
	if d.Desired > 4 {
		t.Fatalf("desired = %d when every registered replica is unhealthy; "+
			"adding machines does not cure a bad checkpoint or a dead "+
			"dependency, it multiplies the failure (reason: %s)",
			d.Desired, d.Reason)
	}
}

func TestNoReplicaCountIsEverBelowFloorOrAboveCeiling(t *testing.T) {
	p := testPolicy()
	p.MinReplicas, p.MaxReplicas = 2, 5
	a, _ := newTestAutoscaler(t, p)

	for waiting := 0; waiting <= 2000; waiting += 37 {
		for ready := 1; ready <= 12; ready++ {
			for _, kv := range []float64{0, 0.5, 0.99} {
				d := a.Decide(ClusterLoad{
					Registered: ready, ReadyReplicas: ready,
					TotalWaiting: waiting, MaxKVUtilization: kv,
				})
				if d.Desired < p.MinReplicas || d.Desired > p.MaxReplicas {
					t.Fatalf("desired %d outside [%d,%d] for ready=%d waiting=%d kv=%.2f",
						d.Desired, p.MinReplicas, p.MaxReplicas, ready, waiting, kv)
				}
			}
		}
	}
}

// TestProcessExecutorStartsAndStops drives the actuator with a trivial
// command, so the loop is exercised without needing a GPU or a checkpoint.
func TestProcessExecutorStartsAndStops(t *testing.T) {
	sleeper := func(index, port int) *exec.Cmd {
		if runtime.GOOS == "windows" {
			return exec.Command("cmd", "/c", "ping -n 60 127.0.0.1 >NUL")
		}
		return exec.Command("sleep", "60")
	}
	ex := &ProcessExecutor{
		Command: sleeper, BasePort: 19100, Log: quietLogger(),
		GraceBeforeKill: 2 * time.Second,
	}
	t.Cleanup(ex.StopAll)

	ctx := context.Background()
	if err := ex.Scale(ctx, 3, "test scale up"); err != nil {
		t.Fatalf("scale up: %v", err)
	}
	if got := ex.Running(); got != 3 {
		t.Fatalf("running = %d after scaling to 3", got)
	}

	if err := ex.Scale(ctx, 1, "test scale down"); err != nil {
		t.Fatalf("scale down: %v", err)
	}
	if got := ex.Running(); got != 1 {
		t.Fatalf("running = %d after scaling to 1", got)
	}

	// Idempotent: applying the same target must not churn processes.
	if err := ex.Scale(ctx, 1, "no-op"); err != nil {
		t.Fatalf("no-op scale: %v", err)
	}
	if got := ex.Running(); got != 1 {
		t.Fatalf("running = %d after a no-op scale", got)
	}
}

// TestRunSkipsTickOnObservationFailure: a broken metrics pipeline must not be
// read as "load is zero". Otherwise a monitoring outage becomes a capacity
// outage, which is the most expensive way for an autoscaler to be wrong.
func TestRunSkipsTickOnObservationFailure(t *testing.T) {
	a, _ := newTestAutoscaler(t, testPolicy())
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	scaled := make(chan int, 4)
	go a.Run(ctx, 10*time.Millisecond,
		collectorFunc(func(context.Context) (ClusterLoad, error) {
			return ClusterLoad{}, context.DeadlineExceeded
		}),
		executorFunc(func(_ context.Context, d int, _ string) error {
			scaled <- d
			return nil
		}),
	)

	select {
	case d := <-scaled:
		t.Fatalf("scaled to %d while observations were failing", d)
	case <-time.After(150 * time.Millisecond):
	}
}

type collectorFunc func(context.Context) (ClusterLoad, error)

func (f collectorFunc) Observe(ctx context.Context) (ClusterLoad, error) { return f(ctx) }

type executorFunc func(context.Context, int, string) error

func (f executorFunc) Scale(ctx context.Context, d int, reason string) error {
	return f(ctx, d, reason)
}
