// Command autoscaler sizes the replica fleet.
//
// Runs as a set: every instance campaigns for leadership in etcd and only the
// winner acts. Standbys sit in Campaign(), so failover costs one lease TTL
// rather than a human noticing.
//
// It observes through the same health RPCs the router uses, so the controller
// and the router never disagree about what the fleet looked like at a given
// instant, and the replicas answer one set of health checks rather than two.
package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/nanoserve/router"
)

func main() {
	var (
		etcdAddr = flag.String("etcd", "127.0.0.1:2379", "comma-separated etcd endpoints")
		id       = flag.String("id", defaultID(), "this controller's identity in the election")
		interval = flag.Duration("interval", 15*time.Second, "control loop period")
		leaseTTL = flag.Int("lease-ttl", 10, "seconds before a dead leader is replaced")

		target    = flag.Float64("target-queue-depth", 4, "waiting requests per replica to steer toward")
		minRep    = flag.Int("min-replicas", 1, "floor")
		maxRep    = flag.Int("max-replicas", 8, "ceiling")
		tolerance = flag.Float64("tolerance", 0.1, "dead band around the target")
		downWin   = flag.Duration("scale-down-stabilization", 5*time.Minute,
			"how long low demand must persist before a replica is given up")
		kvHigh = flag.Float64("kv-high-water", 0.90, "KV utilization that forces headroom")

		mode     = flag.String("executor", "etcd", "etcd (publish desired) or process (run replicas here)")
		basePort = flag.Int("base-port", 9101, "first port for the process executor")
		python   = flag.String("python", "python", "interpreter for spawned replicas")
		checkpt  = flag.String("checkpoint", "checkpoints/run1/best.pt", "model for spawned replicas")
		device   = flag.String("device", "cuda", "device for spawned replicas")
		healthEv = flag.Duration("health-interval", 2*time.Second, "health poll period")
		healthTo = flag.Duration("health-timeout", 1*time.Second, "health poll timeout")
	)
	flag.Parse()

	log := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo}))

	policy := router.ScalePolicy{
		TargetQueueDepth:       *target,
		MinReplicas:            *minRep,
		MaxReplicas:            *maxRep,
		Tolerance:              *tolerance,
		ScaleDownStabilization: *downWin,
		KVHighWater:            *kvHigh,
	}
	scaler, err := router.NewAutoscaler(policy, log)
	if err != nil {
		// Refused at startup rather than at the first tick: a misconfigured
		// policy that only fails later has already been trusted with a
		// production fleet by the time anyone reads the error.
		log.Error("invalid scale policy", "err", err)
		os.Exit(1)
	}

	endpoints := strings.Split(*etcdAddr, ",")
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	// Discovery and health run on every instance, leader or not. A standby
	// that has been watching all along can act the instant it is promoted;
	// one that starts observing only after winning spends its first several
	// ticks with no history, which is exactly the wrong moment to be blind.
	registry, err := router.NewEtcdRegistry(endpoints, log)
	if err != nil {
		log.Error("etcd unavailable", "err", err)
		os.Exit(1)
	}
	defer func() { _ = registry.Close() }()

	events, err := registry.Watch(ctx)
	if err != nil {
		log.Error("registry watch failed", "err", err)
		os.Exit(1)
	}

	ring := router.NewRing(router.DefaultVirtualNodes, router.DefaultEpsilon)
	pool := router.NewClientPool()
	defer func() { _ = pool.Close() }()

	go router.NewReconciler(ring, log).Run(ctx, events)
	go func() {
		ticker := time.NewTicker(*healthEv)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				pool.SyncWith(ring)
			}
		}
	}()

	checker := router.NewHealthChecker(ring, pool, *healthEv, *healthTo, log)
	go checker.Run(ctx)
	collector := router.NewHealthCollector(checker)

	executor, cleanup, err := buildExecutor(*mode, endpoints, *id, *basePort,
		*python, *checkpt, *device, *etcdAddr, log)
	if err != nil {
		log.Error("executor", "err", err)
		os.Exit(1)
	}
	defer cleanup()

	log.Info("autoscaler starting", "id", *id, "executor", *mode,
		"target_queue_depth", *target, "range", fmt.Sprintf("%d-%d", *minRep, *maxRep))

	if err := router.RunAsLeader(ctx, endpoints, *id, *leaseTTL, log,
		func(leaderCtx context.Context) {
			scaler.Run(leaderCtx, *interval, collector, executor)
		},
	); err != nil {
		log.Error("election failed", "err", err)
		os.Exit(1)
	}
	log.Info("autoscaler stopped")
}

func buildExecutor(
	mode string, endpoints []string, id string, basePort int,
	python, checkpoint, device, etcdAddr string, log *slog.Logger,
) (router.Executor, func(), error) {
	switch mode {
	case "etcd":
		ex, err := router.NewEtcdExecutor(endpoints, id, log)
		if err != nil {
			return nil, func() {}, err
		}
		return ex, func() { _ = ex.Close() }, nil

	case "process":
		script := filepath.Join("scripts", "serve_replica.py")
		ex := &router.ProcessExecutor{
			BasePort:        basePort,
			Log:             log,
			GraceBeforeKill: 20 * time.Second,
			Command: func(index, port int) *exec.Cmd {
				cmd := exec.Command(python, "-u", script,
					"--port", fmt.Sprint(port),
					"--replica-id", fmt.Sprintf("replica-%d", index),
					"--checkpoint", checkpoint,
					"--device", device,
					// Spawned replicas register themselves, so the router
					// picks them up with no coordination between the two
					// controllers -- which is the payoff for making the
					// registry the single source of membership truth.
					"--etcd", etcdAddr,
					"--advertise", fmt.Sprintf("127.0.0.1:%d", port),
				)
				cmd.Stdout, cmd.Stderr = os.Stdout, os.Stderr
				return cmd
			},
		}
		return ex, ex.StopAll, nil

	default:
		return nil, func() {}, fmt.Errorf("unknown executor %q (want etcd or process)", mode)
	}
}

func defaultID() string {
	host, err := os.Hostname()
	if err != nil {
		host = "unknown"
	}
	return fmt.Sprintf("%s-%d", host, os.Getpid())
}
