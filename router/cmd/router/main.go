// Command router is the Nanoserve control plane.
//
// Terminates client HTTP connections, picks a replica by prefix hash under a
// load bound, and streams tokens back over SSE. Replicas are discovered from
// a registry and health-checked continuously.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/nanoserve/router"
)

func main() {
	var (
		addr        = flag.String("addr", "127.0.0.1:8080", "HTTP listen address")
		replicas    = flag.String("replicas", "", "comma-separated id=host:port list")
		etcdAddr    = flag.String("etcd", "", "etcd endpoints; empty uses --replicas")
		prefixLen   = flag.Int("prefix-len", 128, "prompt bytes used as the routing key")
		epsilon     = flag.Float64("epsilon", router.DefaultEpsilon, "load bound: cap is (1+eps) x mean")
		vnodes      = flag.Int("vnodes", router.DefaultVirtualNodes, "ring points per replica")
		maxRetries  = flag.Int("max-retries", 2, "replica failovers per request")
		healthEvery = flag.Duration("health-interval", 2*time.Second, "health poll interval")
		healthWait  = flag.Duration("health-timeout", 1*time.Second, "health poll timeout")

		redisAddr = flag.String("redis", "", "Redis for distributed rate limiting; empty disables it")
		rpm       = flag.Float64("rpm", 60, "requests per minute per client")
		tpm       = flag.Float64("tpm", 60000, "generated tokens per minute per client")
		reqBurst  = flag.Float64("request-burst", 10, "request bucket capacity")
		tokBurst  = flag.Float64("token-burst", 10000, "token bucket capacity")
		failOpen  = flag.Bool("ratelimit-fail-open", true,
			"admit requests when Redis is unreachable rather than reject them")
	)
	flag.Parse()

	log := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo}))

	endpoints, err := parseReplicas(*replicas)
	if err != nil {
		log.Error("bad --replicas", "err", err)
		os.Exit(1)
	}
	if len(endpoints) == 0 && *etcdAddr == "" {
		log.Error("no replicas: pass --replicas or --etcd")
		os.Exit(1)
	}

	ring := router.NewRing(*vnodes, *epsilon)
	pool := router.NewClientPool()
	defer pool.Close()

	var registry router.Registry = router.NewStaticRegistry(endpoints)
	if *etcdAddr != "" {
		reg, err := router.NewEtcdRegistry(strings.Split(*etcdAddr, ","), log)
		if err != nil {
			log.Error("etcd unavailable", "err", err)
			os.Exit(1)
		}
		registry = reg
		defer reg.Close()
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	events, err := registry.Watch(ctx)
	if err != nil {
		log.Error("registry watch failed", "err", err)
		os.Exit(1)
	}

	reconciler := router.NewReconciler(ring, log)
	go reconciler.Run(ctx, events)

	// Give the first membership event a moment to land, so the pool is
	// populated before the health checker starts polling names it has no
	// connection for.
	time.Sleep(150 * time.Millisecond)
	pool.SyncWith(ring)

	go func() {
		ticker := time.NewTicker(*healthEvery)
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

	checker := router.NewHealthChecker(ring, pool, *healthEvery, *healthWait, log)
	go checker.Run(ctx)

	proxy := router.NewProxy(ring, pool, *prefixLen, *maxRetries, log)

	if *redisAddr != "" {
		cfg := router.DefaultRateLimitConfig()
		cfg.RequestsPerMinute, cfg.TokensPerMinute = *rpm, *tpm
		cfg.RequestBurst, cfg.TokenBurst = *reqBurst, *tokBurst
		cfg.FailOpen = *failOpen
		limiter, err := router.NewRateLimiter(*redisAddr, cfg)
		if err != nil {
			log.Error("invalid rate limit config", "err", err)
			os.Exit(1)
		}
		defer func() { _ = limiter.Close() }()
		proxy = proxy.WithRateLimiter(limiter)
		log.Info("rate limiting enabled", "redis", *redisAddr,
			"rpm", *rpm, "tpm", *tpm, "fail_open", *failOpen)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("POST /generate", proxy.ServeGenerate)
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, r *http.Request) {
		if ring.ReadyCount() == 0 {
			// 503 while no replica can serve. A router that reports healthy
			// with nothing behind it makes an outage look like a client bug.
			w.WriteHeader(http.StatusServiceUnavailable)
			_, _ = w.Write([]byte(`{"status":"no ready replicas"}`))
			return
		}
		_, _ = w.Write([]byte(`{"status":"ok"}`))
	})
	mux.HandleFunc("GET /stats", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(proxy.Stats())
	})

	srv := &http.Server{
		Addr:    *addr,
		Handler: mux,
		// No write timeout: responses are long-lived token streams, and a
		// write deadline would sever a healthy generation mid-flight.
		ReadHeaderTimeout: 5 * time.Second,
	}

	go func() {
		log.Info("router listening",
			"addr", *addr, "replicas", len(endpoints),
			"epsilon", *epsilon, "prefix_len", *prefixLen)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Error("listen failed", "err", err)
			os.Exit(1)
		}
	}()

	<-ctx.Done()
	log.Info("draining")

	// Graceful shutdown: stop accepting, let in-flight streams finish. The
	// whole point of the drain path is that a deploy costs no dropped
	// requests, and cutting streams here would defeat it.
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if err := srv.Shutdown(shutdownCtx); err != nil {
		log.Warn("shutdown incomplete", "err", err)
	}
	log.Info("stopped")
}

// parseReplicas reads "id=host:port,id2=host:port".
func parseReplicas(spec string) ([]router.Endpoint, error) {
	if strings.TrimSpace(spec) == "" {
		return nil, nil
	}
	var out []router.Endpoint
	for _, part := range strings.Split(spec, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		id, addr, found := strings.Cut(part, "=")
		if !found {
			// Bare host:port is allowed; the address doubles as the id.
			id, addr = part, part
		}
		out = append(out, router.Endpoint{ID: id, Addr: addr})
	}
	return out, nil
}
