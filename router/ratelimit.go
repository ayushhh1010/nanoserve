package router

import (
	"context"
	"errors"
	"fmt"
	"strconv"
	"time"

	"github.com/redis/go-redis/v9"
)

// Distributed rate limiting, in Redis, as a token bucket with variable cost.
//
// Two dimensions, because an LLM endpoint has two scarce resources and they do
// not move together. Requests per minute bounds connection and scheduling
// overhead; *tokens* per minute bounds the GPU. A client sending one request
// that generates 4000 tokens costs the cluster far more than forty requests of
// ten tokens each, and a limiter that counts only requests would wave the
// first one through while throttling the second. This is why the commercial
// APIs quote RPM and TPM separately, and the reason is the same here.
//
// Both dimensions are checked and consumed in a single Lua script, atomically,
// all-or-nothing. Two round trips would be a bug rather than an inefficiency:
// consume from the request bucket, fail the token bucket, and the client has
// been charged for a request it was not allowed to make. Under sustained
// throttling that leak is the difference between a limiter and a slow denial
// of service against your own users.
//
// Time comes from redis.call('TIME'), never from the caller. Routers do not
// share a clock, and NTP skew of even a second across instances lets a client
// aim its burst at whichever router is running slow. The Redis server is the
// one clock every instance already agrees on.

// ErrLimiterUnavailable is returned when Redis cannot be reached.
var ErrLimiterUnavailable = errors.New("router: rate limiter unavailable")

// rateLimitScript checks and consumes two buckets atomically.
//
//	KEYS[1] request bucket   KEYS[2] token bucket
//	ARGV[1] req capacity     ARGV[2] req refill/sec
//	ARGV[3] tok capacity     ARGV[4] tok refill/sec
//	ARGV[5] token cost       ARGV[6] idle TTL seconds
//
// Returns {allowed, req_remaining, tok_remaining, retry_after_seconds}.
// Numbers come back as strings: Lua in Redis truncates numeric returns to
// integers, which would silently floor every fractional token and every
// sub-second retry hint to zero.
const rateLimitScript = `
local now = redis.call('TIME')
local now_s = tonumber(now[1]) + tonumber(now[2]) / 1000000

local req_cap  = tonumber(ARGV[1])
local req_rate = tonumber(ARGV[2])
local tok_cap  = tonumber(ARGV[3])
local tok_rate = tonumber(ARGV[4])
local cost     = tonumber(ARGV[5])
local ttl      = tonumber(ARGV[6])

-- A request larger than the bucket can ever hold is not "try again later",
-- it is "never". Telling such a client to retry produces an infinite polite
-- retry loop that looks exactly like a healthy client from the outside.
if cost > tok_cap then
  return {0, '-1', '-1', '-1'}
end

local function refill(key, cap, rate)
  local state = redis.call('HMGET', key, 'tokens', 'ts')
  local tokens = tonumber(state[1])
  local ts = tonumber(state[2])
  if tokens == nil or ts == nil then
    return cap
  end
  local elapsed = now_s - ts
  if elapsed < 0 then elapsed = 0 end
  local filled = tokens + elapsed * rate
  if filled > cap then filled = cap end
  return filled
end

local req_tokens = refill(KEYS[1], req_cap, req_rate)
local tok_tokens = refill(KEYS[2], tok_cap, tok_rate)

local retry = 0
local allowed = 1
if req_tokens < 1 then
  allowed = 0
  local wait = (1 - req_tokens) / req_rate
  if wait > retry then retry = wait end
end
if tok_tokens < cost then
  allowed = 0
  local wait = (cost - tok_tokens) / tok_rate
  if wait > retry then retry = wait end
end

-- Persist the refill even on rejection. Skipping it would keep 'ts' pinned at
-- the last accepted request, so a throttled client's bucket would appear to
-- refill from an ever-staler timestamp and never recover.
if allowed == 1 then
  req_tokens = req_tokens - 1
  tok_tokens = tok_tokens - cost
end

redis.call('HSET', KEYS[1], 'tokens', req_tokens, 'ts', now_s)
redis.call('HSET', KEYS[2], 'tokens', tok_tokens, 'ts', now_s)
-- Idle buckets expire, so memory tracks active clients rather than every
-- client that has ever appeared. A full bucket is indistinguishable from no
-- bucket, so dropping it loses nothing.
redis.call('PEXPIRE', KEYS[1], math.ceil(ttl * 1000))
redis.call('PEXPIRE', KEYS[2], math.ceil(ttl * 1000))

return {allowed, tostring(req_tokens), tostring(tok_tokens), tostring(retry)}
`

// refundScript returns unused token budget.
//
//	KEYS[1] token bucket
//	ARGV[1] capacity   ARGV[2] refill/sec   ARGV[3] refund   ARGV[4] ttl
const refundScript = `
local now = redis.call('TIME')
local now_s = tonumber(now[1]) + tonumber(now[2]) / 1000000

local cap    = tonumber(ARGV[1])
local rate   = tonumber(ARGV[2])
local refund = tonumber(ARGV[3])
local ttl    = tonumber(ARGV[4])

local state = redis.call('HMGET', KEYS[1], 'tokens', 'ts')
local tokens = tonumber(state[1])
local ts = tonumber(state[2])
if tokens == nil or ts == nil then
  tokens = cap
  ts = now_s
end

local elapsed = now_s - ts
if elapsed < 0 then elapsed = 0 end
tokens = tokens + elapsed * rate + refund
-- Clamped at capacity: a refund restores budget, it must never mint budget
-- the client never had. Without the clamp, repeated reserve/refund cycles
-- would inflate the bucket past its burst and the limit would stop limiting.
if tokens > cap then tokens = cap end

redis.call('HSET', KEYS[1], 'tokens', tokens, 'ts', now_s)
redis.call('PEXPIRE', KEYS[1], math.ceil(ttl * 1000))
return tostring(tokens)
`

// RateLimitConfig describes one tier.
type RateLimitConfig struct {
	// RequestsPerMinute and TokensPerMinute are the sustained rates.
	RequestsPerMinute float64
	TokensPerMinute   float64

	// Burst multipliers over the per-minute rate. A bucket whose capacity
	// equals its per-minute rate allows a full minute of traffic instantly,
	// which is usually not what "10 per minute" is meant to permit.
	RequestBurst float64
	TokenBurst   float64

	// IdleTTL is how long an untouched bucket survives.
	IdleTTL time.Duration

	// FailOpen decides what happens when Redis is unreachable.
	//
	// Defaults to true, and that default is a real decision rather than
	// convenience. Rate limiting is a cost-control and fairness mechanism, not
	// a correctness one: failing closed converts a Redis outage into a total
	// outage of a service that is otherwise perfectly healthy, which trades a
	// billing problem for an availability problem. Set false where the limit
	// is a hard contractual or safety ceiling.
	FailOpen bool
}

func DefaultRateLimitConfig() RateLimitConfig {
	return RateLimitConfig{
		RequestsPerMinute: 60,
		TokensPerMinute:   60_000,
		RequestBurst:      10,
		TokenBurst:        10_000,
		IdleTTL:           10 * time.Minute,
		FailOpen:          true,
	}
}

func (c RateLimitConfig) validate() error {
	switch {
	case c.RequestsPerMinute <= 0:
		return fmt.Errorf("RequestsPerMinute must be > 0, got %v", c.RequestsPerMinute)
	case c.TokensPerMinute <= 0:
		return fmt.Errorf("TokensPerMinute must be > 0, got %v", c.TokensPerMinute)
	case c.RequestBurst < 1:
		return fmt.Errorf("RequestBurst must be >= 1, got %v", c.RequestBurst)
	case c.TokenBurst < 1:
		return fmt.Errorf("TokenBurst must be >= 1, got %v", c.TokenBurst)
	case c.IdleTTL <= 0:
		return fmt.Errorf("IdleTTL must be > 0, got %v", c.IdleTTL)
	}
	return nil
}

// Verdict is the outcome of one admission check.
type Verdict struct {
	Allowed bool
	// RetryAfter is how long until the request could succeed. Zero when
	// allowed; negative when the request can never succeed at this tier,
	// which the caller must report as a permanent error rather than a delay.
	RetryAfter       time.Duration
	RequestRemaining float64
	TokenRemaining   float64
	// Degraded is true when Redis was unreachable and FailOpen let the
	// request through. Surfaced so the caller can count it: silently
	// admitting everything during an outage looks identical to having no
	// traffic problem at all.
	Degraded bool
}

// Permanent reports a request that can never be admitted at this tier.
func (v Verdict) Permanent() bool { return !v.Allowed && v.RetryAfter < 0 }

// RateLimiter enforces per-client limits across every router instance.
type RateLimiter struct {
	rdb    *redis.Client
	cfg    RateLimitConfig
	allow  *redis.Script
	refund *redis.Script
}

func NewRateLimiter(addr string, cfg RateLimitConfig) (*RateLimiter, error) {
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	return &RateLimiter{
		rdb:    redis.NewClient(&redis.Options{Addr: addr}),
		cfg:    cfg,
		allow:  redis.NewScript(rateLimitScript),
		refund: redis.NewScript(refundScript),
	}, nil
}

func (r *RateLimiter) Close() error { return r.rdb.Close() }

// keys returns the two bucket keys for a client.
//
// The hash tag {client} is not decoration: in Redis Cluster a multi-key script
// is rejected outright unless every key hashes to the same slot, so without
// the tag this works on a single node and fails the moment it is deployed on
// the cluster it was designed for.
func (r *RateLimiter) keys(client string) []string {
	return []string{
		fmt.Sprintf("ratelimit:{%s}:req", client),
		fmt.Sprintf("ratelimit:{%s}:tok", client),
	}
}

// Allow reserves budget for a request expected to cost tokenCost tokens.
//
// Reserve-then-refund, because the true cost is unknowable in advance: the
// caller reserves max_tokens up front and returns the difference through
// Refund once generation ends. Charging only on completion would let a client
// open a thousand concurrent 4000-token requests before any of them had been
// billed, which is exactly the burst the limit exists to prevent.
func (r *RateLimiter) Allow(ctx context.Context, client string, tokenCost int) (Verdict, error) {
	cfg := r.cfg
	res, err := r.allow.Run(ctx, r.rdb, r.keys(client),
		cfg.RequestBurst, cfg.RequestsPerMinute/60.0,
		cfg.TokenBurst, cfg.TokensPerMinute/60.0,
		tokenCost, cfg.IdleTTL.Seconds(),
	).Slice()
	if err != nil {
		if cfg.FailOpen {
			return Verdict{Allowed: true, Degraded: true}, nil
		}
		return Verdict{}, fmt.Errorf("%w: %v", ErrLimiterUnavailable, err)
	}
	return parseVerdict(res)
}

// Refund returns unspent token budget after a generation finishes.
func (r *RateLimiter) Refund(ctx context.Context, client string, tokens int) error {
	if tokens <= 0 {
		return nil
	}
	cfg := r.cfg
	_, err := r.refund.Run(ctx, r.rdb, r.keys(client)[1:],
		cfg.TokenBurst, cfg.TokensPerMinute/60.0, tokens, cfg.IdleTTL.Seconds(),
	).Result()
	if err != nil {
		// A lost refund over-charges the client slightly and self-corrects on
		// the next refill. Not worth failing a completed request over.
		return fmt.Errorf("%w: %v", ErrLimiterUnavailable, err)
	}
	return nil
}

func parseVerdict(res []any) (Verdict, error) {
	if len(res) != 4 {
		return Verdict{}, fmt.Errorf("rate limit script returned %d values, want 4", len(res))
	}
	allowed, ok := res[0].(int64)
	if !ok {
		return Verdict{}, fmt.Errorf("rate limit script returned %T for allowed", res[0])
	}
	reqRem, err := parseFloatField(res[1], "req_remaining")
	if err != nil {
		return Verdict{}, err
	}
	tokRem, err := parseFloatField(res[2], "tok_remaining")
	if err != nil {
		return Verdict{}, err
	}
	retry, err := parseFloatField(res[3], "retry_after")
	if err != nil {
		return Verdict{}, err
	}

	v := Verdict{
		Allowed:          allowed == 1,
		RequestRemaining: reqRem,
		TokenRemaining:   tokRem,
	}
	switch {
	case v.Allowed:
		v.RetryAfter = 0
	case retry < 0:
		v.RetryAfter = -1 // never satisfiable at this tier
	default:
		v.RetryAfter = time.Duration(retry * float64(time.Second))
	}
	return v, nil
}

func parseFloatField(v any, name string) (float64, error) {
	s, ok := v.(string)
	if !ok {
		return 0, fmt.Errorf("rate limit script returned %T for %s", v, name)
	}
	f, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return 0, fmt.Errorf("rate limit script returned %q for %s: %w", s, name, err)
	}
	return f, nil
}
