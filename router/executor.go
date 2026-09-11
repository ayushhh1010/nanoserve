package router

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"sync"
	"time"

	clientv3 "go.etcd.io/etcd/client/v3"
)

// DesiredKey is where the autoscaler publishes its decision.
const DesiredKey = "/nanoserve/scale/desired"

// ScaleTarget is the published decision. JSON rather than a bare integer
// because the number alone is unactionable during an incident: the question is
// never "how many replicas" but "why that many, and when was it decided".
type ScaleTarget struct {
	Desired   int       `json:"desired"`
	Reason    string    `json:"reason"`
	DecidedBy string    `json:"decided_by"`
	DecidedAt time.Time `json:"decided_at"`
}

// EtcdExecutor publishes the desired count and lets something else apply it.
//
// This is the split every real autoscaler has: the controller decides, and a
// separate actuator with the credentials to create machines acts. Keeping them
// apart means the decision is auditable and replayable on its own, and that a
// bug in the actuator cannot corrupt the policy. In Kubernetes the actuator is
// the Deployment controller reading .spec.replicas; here it is whatever reads
// this key.
type EtcdExecutor struct {
	client *clientv3.Client
	id     string
	log    *slog.Logger
}

func NewEtcdExecutor(endpoints []string, id string, log *slog.Logger) (*EtcdExecutor, error) {
	client, err := clientv3.New(clientv3.Config{
		Endpoints:   endpoints,
		DialTimeout: 5 * time.Second,
	})
	if err != nil {
		return nil, fmt.Errorf("etcd connect: %w", err)
	}
	return &EtcdExecutor{client: client, id: id, log: log}, nil
}

func (e *EtcdExecutor) Scale(ctx context.Context, desired int, reason string) error {
	payload, err := json.Marshal(ScaleTarget{
		Desired: desired, Reason: reason,
		DecidedBy: e.id, DecidedAt: time.Now().UTC(),
	})
	if err != nil {
		return fmt.Errorf("encode scale target: %w", err)
	}
	cctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	if _, err := e.client.Put(cctx, DesiredKey, string(payload)); err != nil {
		return fmt.Errorf("publish desired=%d: %w", desired, err)
	}
	return nil
}

func (e *EtcdExecutor) Close() error { return e.client.Close() }

// ReadDesired returns the currently published target, if any.
func ReadDesired(ctx context.Context, client *clientv3.Client) (*ScaleTarget, error) {
	resp, err := client.Get(ctx, DesiredKey)
	if err != nil {
		return nil, err
	}
	if len(resp.Kvs) == 0 {
		return nil, nil
	}
	var t ScaleTarget
	if err := json.Unmarshal(resp.Kvs[0].Value, &t); err != nil {
		return nil, fmt.Errorf("decode scale target: %w", err)
	}
	return &t, nil
}

// ProcessExecutor starts and stops replica processes on this machine.
//
// The actuator for a single-box demo, where "provision a node" means "run
// another Python process". It closes the loop so the autoscaler can be shown
// actually working rather than only logging intentions.
//
// Scale-down stops the most recently started replica first. Newest-first is
// not arbitrary: older replicas have had longer to accumulate a warm prefix
// cache, so they are the expensive ones to discard -- the same reasoning that
// makes the stabilization window asymmetric in the first place.
type ProcessExecutor struct {
	// Command builds the argv for replica i. Injected so tests can scale a
	// trivial process instead of a model server.
	Command  func(index int, port int) *exec.Cmd
	BasePort int
	Log      *slog.Logger

	// GraceBeforeKill is how long a stopped replica has to drain. It gets a
	// SIGTERM-equivalent first so it can deregister and finish in-flight work;
	// killing outright would drop live generations on every scale-down, which
	// would make autoscaling strictly worse than not autoscaling.
	GraceBeforeKill time.Duration

	mu      sync.Mutex
	running []*replicaProc
	nextIdx int
}

type replicaProc struct {
	index int
	port  int
	cmd   *exec.Cmd
}

func (p *ProcessExecutor) Scale(ctx context.Context, desired int, reason string) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	for len(p.running) < desired {
		port := p.BasePort + p.nextIdx
		cmd := p.Command(p.nextIdx, port)
		if err := cmd.Start(); err != nil {
			return fmt.Errorf("start replica %d: %w", p.nextIdx, err)
		}
		p.log().Info("started replica", "index", p.nextIdx, "port", port,
			"pid", cmd.Process.Pid, "reason", reason)
		p.running = append(p.running, &replicaProc{index: p.nextIdx, port: port, cmd: cmd})
		p.nextIdx++
	}

	for len(p.running) > desired && len(p.running) > 0 {
		victim := p.running[len(p.running)-1]
		p.running = p.running[:len(p.running)-1]
		p.log().Info("stopping replica", "index", victim.index, "port", victim.port,
			"pid", victim.cmd.Process.Pid, "reason", reason)
		p.terminate(victim)
	}
	return nil
}

// Running reports how many replica processes this executor has up.
func (p *ProcessExecutor) Running() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return len(p.running)
}

// StopAll terminates every replica. For shutdown and for tests.
func (p *ProcessExecutor) StopAll() {
	p.mu.Lock()
	victims := p.running
	p.running = nil
	p.mu.Unlock()
	for _, v := range victims {
		p.terminate(v)
	}
}

func (p *ProcessExecutor) terminate(v *replicaProc) {
	grace := p.GraceBeforeKill
	if grace <= 0 {
		grace = 15 * time.Second
	}
	// os.Interrupt is not implemented on Windows, so signalling is
	// best-effort; Kill after the grace period is the guaranteed path either
	// way, and the replica's own drain handler is what makes the graceful
	// case graceful.
	if err := v.cmd.Process.Signal(os.Interrupt); err != nil {
		p.log().Debug("interrupt not delivered; will kill after grace",
			"index", v.index, "err", err)
	}
	done := make(chan struct{})
	go func() {
		_, _ = v.cmd.Process.Wait()
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(grace):
		p.log().Warn("replica did not exit within grace; killing", "index", v.index)
		_ = v.cmd.Process.Kill()
		<-done
	}
}

func (p *ProcessExecutor) log() *slog.Logger {
	if p.Log != nil {
		return p.Log
	}
	return slog.Default()
}

// healthCollector adapts the HealthChecker to the Collector interface.
type healthCollector struct{ checker *HealthChecker }

// NewHealthCollector observes the fleet through the health checker the router
// already runs.
func NewHealthCollector(checker *HealthChecker) Collector {
	return &healthCollector{checker: checker}
}

func (h *healthCollector) Observe(ctx context.Context) (ClusterLoad, error) {
	// Reported as observed, zeros included. An earlier version returned an
	// error whenever nothing was ready, which was worse than it looked: a
	// fleet that had entirely died would make every tick a skipped tick, so
	// the one situation most needing a scale-up was the one situation the
	// autoscaler refused to act on. Which zero this is -- empty registry
	// versus registered-but-unhealthy -- is carried in Registered, and it is
	// the policy's job to tell them apart, not the collector's.
	return h.checker.Load(), nil
}
