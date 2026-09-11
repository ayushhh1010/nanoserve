package router

import (
	"context"
	"fmt"
	"log/slog"
	"math"
	"sync"
	"time"
)

// Autoscaling for an inference cluster, modelled on the Kubernetes HPA
// algorithm because the failure modes it guards against are the same ones.
//
//	desired = ceil(current * observed / target)
//
// with two guards that exist for reasons worth stating:
//
// **Tolerance.** A metric that sits at 1.03x target should not cause a scaling
// event. Without a dead band the controller reacts to noise, and every
// reaction changes the metric, which is the definition of an oscillator. HPA
// uses 10%; so does this.
//
// **Asymmetric stabilization.** Scaling up is cheap and urgent: the queue is
// already growing. Scaling down is neither. HPA's defaults -- act immediately
// on scale-up, require 5 minutes of sustained evidence to scale down -- are
// the right shape here and more so than for a stateless web service, because
// a replica is not fungible: it holds a warm KV prefix cache that took real
// prefill work to build, and killing it moves every key it owned to another
// replica, which must then prefill them again. A scale-down that turns out to
// be premature costs a latency spike on the way down *and* another on the way
// back up.
//
// The metric is queue depth per replica rather than CPU or GPU utilization. A
// GPU that is busy is not a GPU that is behind -- a saturated decode loop at
// 100% utilization with an empty queue is exactly what you want, and scaling
// on it would add replicas forever. Waiting requests are what a user feels.
//
// KV utilization is a separate, non-proportional guard. It is the real
// capacity limit of a serving replica and it does not show up in queue depth
// until the moment it does, all at once: a replica at 97% KV is one long
// prompt away from preempting sequences, and by the time that reaches the
// queue the damage is already taken.

// ScalePolicy is the autoscaler's configuration.
type ScalePolicy struct {
	// TargetQueueDepth is the waiting-request count per replica the
	// controller steers toward. Not zero: a queue that is always empty means
	// capacity is sitting idle, and continuous batching needs a few queued
	// requests to fill a batch efficiently.
	TargetQueueDepth float64

	MinReplicas int
	MaxReplicas int

	// Tolerance is the dead band around the target, as a fraction. HPA's
	// default is 0.1 and the reasoning carries over unchanged.
	Tolerance float64

	// ScaleDownStabilization is how long the desired count must stay low
	// before the controller acts on it. Scale-up has no equivalent window on
	// purpose -- see the asymmetry note above.
	ScaleDownStabilization time.Duration

	// KVHighWater forces a scale-up regardless of queue depth. Fraction 0-1.
	KVHighWater float64
}

func DefaultScalePolicy() ScalePolicy {
	return ScalePolicy{
		TargetQueueDepth:       4.0,
		MinReplicas:            1,
		MaxReplicas:            8,
		Tolerance:              0.1,
		ScaleDownStabilization: 5 * time.Minute,
		KVHighWater:            0.90,
	}
}

func (p ScalePolicy) validate() error {
	switch {
	case p.TargetQueueDepth <= 0:
		return fmt.Errorf("TargetQueueDepth must be > 0, got %v", p.TargetQueueDepth)
	case p.MinReplicas < 1:
		return fmt.Errorf("MinReplicas must be >= 1, got %d", p.MinReplicas)
	case p.MaxReplicas < p.MinReplicas:
		return fmt.Errorf("MaxReplicas (%d) < MinReplicas (%d)",
			p.MaxReplicas, p.MinReplicas)
	case p.Tolerance < 0 || p.Tolerance >= 1:
		return fmt.Errorf("Tolerance must be in [0,1), got %v", p.Tolerance)
	case p.KVHighWater <= 0 || p.KVHighWater > 1:
		return fmt.Errorf("KVHighWater must be in (0,1], got %v", p.KVHighWater)
	}
	return nil
}

// ClusterLoad is one observation of the fleet.
type ClusterLoad struct {
	// Registered is how many replicas exist in the registry, healthy or not.
	// Kept separate from ReadyReplicas because the two zeros mean opposite
	// things: nothing registered means the fleet is gone and must be built,
	// while replicas registered but none ready means they exist and are sick
	// -- and adding more machines does not cure sick ones.
	Registered    int
	ReadyReplicas int
	TotalWaiting  int
	TotalRunning  int
	// MaxKVUtilization is the worst replica, not the mean. A mean hides the
	// one replica that is about to start preempting, and that replica is the
	// one that will produce the latency outlier.
	MaxKVUtilization float64
}

// Decision is what the controller concluded, and why.
type Decision struct {
	Desired int
	Current int
	Reason  string
	// Held is true when a scale-down was computed but suppressed by the
	// stabilization window. Surfaced so the logs distinguish "no change
	// needed" from "change wanted, waiting for evidence" -- two very
	// different things to read at 3am.
	Held bool
}

func (d Decision) Changed() bool { return d.Desired != d.Current }

// Autoscaler holds the stabilization history between observations.
type Autoscaler struct {
	policy ScalePolicy
	log    *slog.Logger

	mu sync.Mutex
	// history of recent raw desired counts, for the scale-down window.
	history []sample
	now     func() time.Time // injectable so tests need not sleep
}

type sample struct {
	at      time.Time
	desired int
}

func NewAutoscaler(policy ScalePolicy, log *slog.Logger) (*Autoscaler, error) {
	if err := policy.validate(); err != nil {
		return nil, err
	}
	return &Autoscaler{policy: policy, log: log, now: time.Now}, nil
}

// Decide computes the replica count for one observation.
func (a *Autoscaler) Decide(load ClusterLoad) Decision {
	a.mu.Lock()
	defer a.mu.Unlock()

	p := a.policy
	current := load.ReadyReplicas

	// Nothing ready. Dividing by zero replicas to get a per-replica metric
	// would give +Inf and slam straight into MaxReplicas, which is the worst
	// available response to a cold start.
	if current <= 0 {
		if load.Registered > 0 {
			// Replicas exist and none are healthy. This is an outage, not a
			// capacity shortage, and the two want opposite responses: adding
			// replicas to a fleet that is failing for its own reasons -- a bad
			// checkpoint, a full disk, a dependency down -- multiplies the
			// failure and burns the budget that recovery needs. Hold.
			return Decision{
				Desired: load.Registered, Current: current,
				Reason: fmt.Sprintf("%d registered, 0 ready: an outage, not a "+
					"capacity shortage; holding", load.Registered),
			}
		}
		return Decision{
			Desired: p.MinReplicas, Current: current,
			Reason: "registry empty; requesting the floor",
		}
	}

	observed := float64(load.TotalWaiting) / float64(current)
	ratio := observed / p.TargetQueueDepth

	raw := current
	reason := fmt.Sprintf("queue %.2f/replica within %.0f%% of target %.2f",
		observed, p.Tolerance*100, p.TargetQueueDepth)

	if math.Abs(ratio-1.0) > p.Tolerance {
		raw = int(math.Ceil(float64(current) * ratio))
		reason = fmt.Sprintf("queue %.2f/replica vs target %.2f (ratio %.2f)",
			observed, p.TargetQueueDepth, ratio)
	}

	// KV pressure overrides the proportional result upward, never downward.
	// A replica can be short of KV while its queue is empty -- a few very long
	// sequences -- and that is precisely when adding capacity is cheapest,
	// before preemption starts recomputing work.
	if load.MaxKVUtilization >= p.KVHighWater && raw <= current {
		raw = current + 1
		reason = fmt.Sprintf("KV at %.0f%% (high water %.0f%%); adding headroom "+
			"before preemption starts", load.MaxKVUtilization*100, p.KVHighWater*100)
	}

	raw = clamp(raw, p.MinReplicas, p.MaxReplicas)

	// Record before stabilizing, so the window sees raw intent.
	now := a.now()
	a.history = append(a.history, sample{at: now, desired: raw})
	a.trim(now)

	if raw >= current {
		// Scale up, or hold. Immediate: the queue is already built.
		return Decision{Desired: raw, Current: current, Reason: reason}
	}

	// Scale down. Use the highest desired count seen in the window, which is
	// HPA's rule and the conservative one: a single quiet moment inside a busy
	// five minutes must not be enough to give up a warm replica.
	stabilized := raw
	for _, s := range a.history {
		if s.desired > stabilized {
			stabilized = s.desired
		}
	}
	if stabilized >= current {
		return Decision{
			Desired: current, Current: current, Held: true,
			Reason: fmt.Sprintf("%s; scale-down to %d held, %v window still "+
				"contains a demand for %d", reason, raw,
				p.ScaleDownStabilization, stabilized),
		}
	}
	return Decision{
		Desired: stabilized, Current: current,
		Reason: fmt.Sprintf("%s; sustained for %v", reason, p.ScaleDownStabilization),
	}
}

func (a *Autoscaler) trim(now time.Time) {
	cutoff := now.Add(-a.policy.ScaleDownStabilization)
	i := 0
	for i < len(a.history) && a.history[i].at.Before(cutoff) {
		i++
	}
	a.history = a.history[i:]
}

func clamp(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// Executor applies a decision. Separated from the decision so the policy can
// be tested exhaustively without anything being started or killed, and so the
// same controller drives a local process supervisor, an etcd key a Kubernetes
// operator reads, or a real Deployment scale call.
type Executor interface {
	Scale(ctx context.Context, desired int, reason string) error
}

// Collector supplies observations.
type Collector interface {
	Observe(ctx context.Context) (ClusterLoad, error)
}

// Run drives the control loop until ctx ends. Caller guarantees leadership.
func (a *Autoscaler) Run(
	ctx context.Context, interval time.Duration, c Collector, e Executor,
) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			load, err := c.Observe(ctx)
			if err != nil {
				// An observation failure must not be read as "load is zero".
				// Scaling down because the metrics pipeline broke is how a
				// monitoring outage becomes a capacity outage.
				a.log.Warn("skipping tick: cannot observe cluster", "err", err)
				continue
			}
			d := a.Decide(load)
			switch {
			case d.Changed():
				a.log.Info("scaling", "from", d.Current, "to", d.Desired,
					"reason", d.Reason)
				if err := e.Scale(ctx, d.Desired, d.Reason); err != nil {
					a.log.Error("scale failed", "desired", d.Desired, "err", err)
				}
			case d.Held:
				a.log.Info("scale-down held", "reason", d.Reason)
			default:
				a.log.Debug("steady", "replicas", d.Current, "reason", d.Reason)
			}
		}
	}
}
