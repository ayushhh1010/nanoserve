package router

import (
	"context"
	"fmt"
	"os"
	"sync"
	"testing"
	"time"
)

// Against a real Redis, because the whole design rests on Lua atomicity and
// on redis.call('TIME'). A mock would be asserting that the mock is atomic.
//
//	docker run -d --name nanoserve-redis -p 6379:6379 redis:8-alpine
//	NANOSERVE_REDIS=127.0.0.1:6379 go test ./... -run RateLimit -v
func redisAddr(t *testing.T) string {
	t.Helper()
	addr := os.Getenv("NANOSERVE_REDIS")
	if addr == "" {
		t.Skip("set NANOSERVE_REDIS=host:port to run rate limiter tests")
	}
	return addr
}

func newLimiter(t *testing.T, cfg RateLimitConfig) (*RateLimiter, string) {
	t.Helper()
	rl, err := NewRateLimiter(redisAddr(t), cfg)
	if err != nil {
		t.Fatalf("NewRateLimiter: %v", err)
	}
	t.Cleanup(func() { _ = rl.Close() })
	// A unique client per test, so runs never inherit another test's bucket.
	client := fmt.Sprintf("test-%s-%d", t.Name(), time.Now().UnixNano())
	return rl, client
}

func TestRateLimitConfigValidation(t *testing.T) {
	cases := map[string]func(*RateLimitConfig){
		"no requests": func(c *RateLimitConfig) { c.RequestsPerMinute = 0 },
		"no tokens":   func(c *RateLimitConfig) { c.TokensPerMinute = 0 },
		"burst < 1":   func(c *RateLimitConfig) { c.RequestBurst = 0 },
		"no ttl":      func(c *RateLimitConfig) { c.IdleTTL = 0 },
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			cfg := DefaultRateLimitConfig()
			mutate(&cfg)
			if _, err := NewRateLimiter("127.0.0.1:6379", cfg); err == nil {
				t.Fatal("expected an invalid config to be rejected at construction")
			}
		})
	}
}

func TestRateLimitAllowsBurstThenThrottles(t *testing.T) {
	cfg := DefaultRateLimitConfig()
	cfg.RequestsPerMinute = 60 // 1/sec
	cfg.RequestBurst = 5
	cfg.TokensPerMinute = 1e9 // effectively unlimited, isolating the request axis
	cfg.TokenBurst = 1e9
	rl, client := newLimiter(t, cfg)
	ctx := context.Background()

	for i := 0; i < 5; i++ {
		v, err := rl.Allow(ctx, client, 1)
		if err != nil {
			t.Fatalf("Allow %d: %v", i, err)
		}
		if !v.Allowed {
			t.Fatalf("request %d of a burst of 5 was rejected (remaining %.2f)",
				i, v.RequestRemaining)
		}
	}

	v, err := rl.Allow(ctx, client, 1)
	if err != nil {
		t.Fatalf("Allow: %v", err)
	}
	if v.Allowed {
		t.Fatal("the 6th request in a burst of 5 was admitted")
	}
	if v.RetryAfter <= 0 || v.RetryAfter > 2*time.Second {
		t.Fatalf("RetryAfter = %v, want roughly 1s at 1 request/sec; a wrong "+
			"hint is worse than none -- clients obey it", v.RetryAfter)
	}
}

// TestRateLimitCountsTokensNotJustRequests is the point of the two-dimensional
// design: one huge request must be throttled even though it is only one
// request.
func TestRateLimitCountsTokensNotJustRequests(t *testing.T) {
	cfg := DefaultRateLimitConfig()
	cfg.RequestsPerMinute = 6000 // not the binding constraint
	cfg.RequestBurst = 1000
	cfg.TokensPerMinute = 600
	cfg.TokenBurst = 1000
	rl, client := newLimiter(t, cfg)
	ctx := context.Background()

	v, err := rl.Allow(ctx, client, 900)
	if err != nil {
		t.Fatalf("Allow: %v", err)
	}
	if !v.Allowed {
		t.Fatalf("a 900-token request against a 1000-token burst was rejected")
	}

	// Only ~100 tokens left. A second large request must not pass, even
	// though the request-per-minute budget is nowhere near exhausted.
	v, err = rl.Allow(ctx, client, 900)
	if err != nil {
		t.Fatalf("Allow: %v", err)
	}
	if v.Allowed {
		t.Fatal("a second 900-token request was admitted with ~100 tokens of " +
			"budget left; counting requests alone lets one client monopolise " +
			"the GPU while staying inside its request limit")
	}
}

// TestRateLimitRejectsImpossibleCostPermanently: a request bigger than the
// bucket will never fit. Telling that client to retry produces an infinite
// polite retry loop that is indistinguishable from healthy traffic.
func TestRateLimitRejectsImpossibleCostPermanently(t *testing.T) {
	cfg := DefaultRateLimitConfig()
	cfg.TokenBurst = 100
	cfg.TokensPerMinute = 100
	rl, client := newLimiter(t, cfg)

	v, err := rl.Allow(context.Background(), client, 5000)
	if err != nil {
		t.Fatalf("Allow: %v", err)
	}
	if v.Allowed {
		t.Fatal("a request larger than the bucket capacity was admitted")
	}
	if !v.Permanent() {
		t.Fatalf("RetryAfter = %v for a request that can never fit; this must "+
			"be a permanent error, not a delay", v.RetryAfter)
	}
}

// TestRateLimitRejectionDoesNotConsumeBudget: a throttled request must not be
// charged. Otherwise a client that keeps retrying can never recover, because
// every rejected attempt pushes the recovery further away.
func TestRateLimitRejectionDoesNotConsumeBudget(t *testing.T) {
	cfg := DefaultRateLimitConfig()
	cfg.RequestsPerMinute = 60
	cfg.RequestBurst = 2
	cfg.TokensPerMinute = 6000
	cfg.TokenBurst = 100
	rl, client := newLimiter(t, cfg)
	ctx := context.Background()

	for i := 0; i < 2; i++ {
		if v, _ := rl.Allow(ctx, client, 1); !v.Allowed {
			t.Fatalf("burst request %d rejected", i)
		}
	}

	var after float64
	for i := 0; i < 5; i++ {
		v, err := rl.Allow(ctx, client, 1)
		if err != nil {
			t.Fatalf("Allow: %v", err)
		}
		if v.Allowed {
			t.Fatal("admitted past the burst without waiting")
		}
		after = v.TokenRemaining
	}
	// Five rejected attempts must not have spent five tokens of budget.
	if after < 90 {
		t.Fatalf("token budget fell to %.1f after five *rejected* requests; "+
			"rejections must not be charged or a retrying client can never "+
			"recover", after)
	}
}

// TestRateLimitRefundReturnsUnusedBudget covers reserve-then-refund: the
// caller reserves max_tokens because the true length is unknowable up front.
func TestRateLimitRefundReturnsUnusedBudget(t *testing.T) {
	cfg := DefaultRateLimitConfig()
	cfg.RequestsPerMinute = 6000
	cfg.RequestBurst = 1000
	cfg.TokensPerMinute = 60 // 1/sec: slow enough that refill cannot explain the gain
	cfg.TokenBurst = 1000
	rl, client := newLimiter(t, cfg)
	ctx := context.Background()

	v, err := rl.Allow(ctx, client, 800)
	if err != nil || !v.Allowed {
		t.Fatalf("reserve: allowed=%v err=%v", v.Allowed, err)
	}
	reserved := v.TokenRemaining

	// The generation actually produced 50 tokens, so 750 come back.
	if err := rl.Refund(ctx, client, 750); err != nil {
		t.Fatalf("Refund: %v", err)
	}

	v2, err := rl.Allow(ctx, client, 1)
	if err != nil {
		t.Fatalf("Allow after refund: %v", err)
	}
	if v2.TokenRemaining < reserved+700 {
		t.Fatalf("remaining %.1f after refunding 750 (was %.1f); without the "+
			"refund, reserving max_tokens would bill every client for output "+
			"it never received", v2.TokenRemaining, reserved)
	}
}

// TestRateLimitRefundCannotMintBudget: the clamp at capacity. Without it,
// repeated reserve/refund cycles inflate the bucket past its burst and the
// limit quietly stops limiting.
func TestRateLimitRefundCannotMintBudget(t *testing.T) {
	cfg := DefaultRateLimitConfig()
	cfg.TokensPerMinute = 600
	cfg.TokenBurst = 100
	cfg.RequestsPerMinute = 6000
	cfg.RequestBurst = 1000
	rl, client := newLimiter(t, cfg)
	ctx := context.Background()

	for i := 0; i < 20; i++ {
		if err := rl.Refund(ctx, client, 1000); err != nil {
			t.Fatalf("Refund: %v", err)
		}
	}
	v, err := rl.Allow(ctx, client, 1)
	if err != nil {
		t.Fatalf("Allow: %v", err)
	}
	if v.TokenRemaining > cfg.TokenBurst {
		t.Fatalf("bucket holds %.1f tokens after refunds, capacity is %.1f; "+
			"a refund restores budget, it must never create it",
			v.TokenRemaining, cfg.TokenBurst)
	}
}

// TestRateLimitIsAtomicUnderConcurrency is the reason this lives in Lua.
// Twenty goroutines racing on a burst of 10 must see exactly 10 admissions.
// A read-modify-write from the client would over-admit here, and would do so
// only under the load that makes the limit matter.
func TestRateLimitIsAtomicUnderConcurrency(t *testing.T) {
	cfg := DefaultRateLimitConfig()
	cfg.RequestsPerMinute = 6 // 0.1/sec: refill cannot mask over-admission
	cfg.RequestBurst = 10
	cfg.TokensPerMinute = 1e9
	cfg.TokenBurst = 1e9
	rl, client := newLimiter(t, cfg)

	const racers = 60
	var (
		wg      sync.WaitGroup
		mu      sync.Mutex
		granted int
	)
	start := make(chan struct{})
	for i := 0; i < racers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			v, err := rl.Allow(context.Background(), client, 1)
			if err != nil {
				return
			}
			if v.Allowed {
				mu.Lock()
				granted++
				mu.Unlock()
			}
		}()
	}
	close(start)
	wg.Wait()

	if granted != 10 {
		t.Fatalf("%d of %d concurrent requests admitted against a burst of 10; "+
			"the check-and-consume must be one atomic script, and this is the "+
			"load at which a non-atomic limiter fails", granted, racers)
	}
}

// TestRateLimitFailsOpenWhenRedisIsGone: a limiter outage must not become a
// service outage. Fairness and cost control are worth less than availability.
func TestRateLimitFailsOpenWhenRedisIsGone(t *testing.T) {
	cfg := DefaultRateLimitConfig()
	cfg.FailOpen = true
	rl, err := NewRateLimiter("127.0.0.1:1", cfg) // nothing listens here
	if err != nil {
		t.Fatalf("NewRateLimiter: %v", err)
	}
	t.Cleanup(func() { _ = rl.Close() })

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	v, err := rl.Allow(ctx, "anyone", 1)
	if err != nil {
		t.Fatalf("fail-open must not surface an error: %v", err)
	}
	if !v.Allowed {
		t.Fatal("request rejected while Redis was unreachable; a rate limiter " +
			"outage would take down a healthy service")
	}
	if !v.Degraded {
		t.Fatal("a fail-open admission must be flagged Degraded, or an outage " +
			"is indistinguishable from having no traffic problem")
	}
}

func TestRateLimitFailsClosedWhenConfigured(t *testing.T) {
	cfg := DefaultRateLimitConfig()
	cfg.FailOpen = false
	rl, err := NewRateLimiter("127.0.0.1:1", cfg)
	if err != nil {
		t.Fatalf("NewRateLimiter: %v", err)
	}
	t.Cleanup(func() { _ = rl.Close() })

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if _, err := rl.Allow(ctx, "anyone", 1); err == nil {
		t.Fatal("fail-closed must surface the error so the caller can reject")
	}
}
