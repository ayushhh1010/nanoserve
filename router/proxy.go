package router

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"net"
	"net/http"
	"strconv"
	"sync/atomic"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// Proxy terminates client HTTP connections and streams from a replica.
//
// The retry path is the interesting part. When a replica dies mid-generation,
// the client has already received some tokens. Restarting from scratch would
// send them again -- visible, wrong, and worse than the failure. So the router
// tracks what it has forwarded and replays prompt + delivered tokens to the
// new replica as resume_tokens, which the replica treats as context rather
// than as output. The client sees one unbroken stream.
//
// Retries carry the same request_id. A replica that is merely slow rather than
// dead may still be running the original, and the idempotency key is what
// stops the cluster doing the work twice at the moment it can least afford it.
type Proxy struct {
	ring       *Ring
	pool       *ClientPool
	log        *slog.Logger
	prefixLen  int
	maxRetries int
	// reconciler is optional, used only to surface degraded membership in
	// /stats. The proxy never asks it to make a routing decision.
	reconciler *Reconciler
	// metrics is optional; nil means the counters below are the only record.
	metrics *Metrics
	// limiter is optional. Nil means no rate limiting, which is a legitimate
	// single-tenant deployment rather than a missing feature.
	limiter *RateLimiter

	// Counters for /metrics. Atomic rather than mutex-guarded: they are
	// touched on every request and never read in the same critical section as
	// anything else.
	requests   atomic.Int64
	retries    atomic.Int64
	migrations atomic.Int64
	failures   atomic.Int64
	rejected   atomic.Int64
	throttled  atomic.Int64
	degraded   atomic.Int64
}

// WithReconciler lets /stats report replicas being retained against the
// registry's wishes -- a degraded state that is otherwise only in the logs.
func (p *Proxy) WithReconciler(rc *Reconciler) *Proxy {
	p.reconciler = rc
	return p
}

// WithMetrics attaches the Prometheus surface.
func (p *Proxy) WithMetrics(m *Metrics) *Proxy {
	p.metrics = m
	return p
}

// WithRateLimiter enables per-client limits. Optional at construction so the
// router runs unchanged without Redis.
func (p *Proxy) WithRateLimiter(rl *RateLimiter) *Proxy {
	p.limiter = rl
	return p
}

func NewProxy(ring *Ring, pool *ClientPool, prefixLen, maxRetries int, log *slog.Logger) *Proxy {
	return &Proxy{
		ring: ring, pool: pool, log: log,
		prefixLen: prefixLen, maxRetries: maxRetries,
	}
}

type generateBody struct {
	Prompt      string  `json:"prompt"`
	MaxTokens   uint32  `json:"max_tokens"`
	Temperature float32 `json:"temperature"`
	TopK        uint32  `json:"top_k"`
	TopP        float32 `json:"top_p"`
	IgnoreEOS   bool    `json:"ignore_eos"`
	SLOSeconds  float64 `json:"slo_seconds"`
	RequestID   string  `json:"request_id"`
}

// ServeGenerate handles POST /generate with an SSE response.
func (p *Proxy) ServeGenerate(w http.ResponseWriter, r *http.Request) {
	var body generateBody
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		http.Error(w, `{"error":"invalid json"}`, http.StatusBadRequest)
		return
	}
	if body.Prompt == "" {
		http.Error(w, `{"error":"prompt is required"}`, http.StatusBadRequest)
		return
	}
	if body.MaxTokens == 0 {
		body.MaxTokens = 128
	}
	if body.RequestID == "" {
		body.RequestID = fmt.Sprintf("r-%d", time.Now().UnixNano())
	}

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, `{"error":"streaming unsupported"}`, http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Request-Id", body.RequestID)

	// Limit before dispatching, not after. The point of a limit is to refuse
	// work before it costs anything; checking once a replica is already
	// prefilling would bill the GPU for every rejected request.
	//
	// Reserve MaxTokens, refund the remainder when the stream ends. The true
	// cost is unknowable up front, and charging only on completion would let a
	// client open a thousand concurrent maximum-length generations before any
	// of them had been billed.
	client := clientKey(r)
	if p.limiter != nil {
		v, err := p.limiter.Allow(r.Context(), client, int(body.MaxTokens))
		if err != nil {
			p.observeThrottle("limiter_unavailable")
			http.Error(w, `{"error":"rate limiter unavailable"}`,
				http.StatusServiceUnavailable)
			return
		}
		if v.Degraded {
			p.degraded.Add(1)
			if p.metrics != nil {
				p.metrics.Degraded.Inc()
			}
		}
		if !v.Allowed {
			p.throttled.Add(1)
			// 429 with Retry-After, except where no amount of waiting helps:
			// a request larger than the client tier can ever admit is a
			// permanent 413, and telling it to retry would produce a polite
			// infinite loop indistinguishable from healthy traffic.
			if v.Permanent() {
				p.observeThrottle("oversized")
				http.Error(w, `{"error":"max_tokens exceeds this tier's burst limit"}`,
					http.StatusRequestEntityTooLarge)
				return
			}
			p.observeThrottle("rate")
			w.Header().Set("Retry-After",
				strconv.Itoa(int(math.Ceil(v.RetryAfter.Seconds()))))
			http.Error(w, `{"error":"rate limit exceeded"}`, http.StatusTooManyRequests)
			return
		}
	}

	p.requests.Add(1)
	startedAt := time.Now()
	delivered := 0
	if p.limiter != nil {
		defer func() {
			// Refund on every exit path, including client disconnects and
			// errors: an abandoned stream that keeps its full reservation
			// charges the client for tokens nobody ever generated.
			if unused := int(body.MaxTokens) - delivered; unused > 0 {
				ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
				defer cancel()
				if err := p.limiter.Refund(ctx, client, unused); err != nil {
					p.log.Warn("refund failed; client over-charged until refill",
						"client", client, "tokens", unused, "err", err)
				}
			}
		}()
	}

	if err := p.stream(r.Context(), w, flusher, body, &delivered); err != nil {
		p.failures.Add(1)
		p.observeOutcome("error", 0)
		p.log.Error("request failed", "request_id", body.RequestID, "err", err)
		writeEvent(w, flusher, "error", map[string]string{"error": err.Error()})
		return
	}
	p.observeOutcome("success", time.Since(startedAt))
}

func (p *Proxy) observeThrottle(reason string) {
	p.throttled.Add(1)
	if p.metrics != nil {
		p.metrics.Throttled.WithLabelValues(reason).Inc()
		p.metrics.Requests.WithLabelValues("throttled").Inc()
	}
}

func (p *Proxy) observeOutcome(outcome string, d time.Duration) {
	if p.metrics == nil {
		return
	}
	p.metrics.Requests.WithLabelValues(outcome).Inc()
	// Duration on success only. A request that failed in 3ms is not a fast
	// request, and folding it in drags the latency distribution toward zero
	// exactly when the system is least healthy.
	if outcome == "success" {
		p.metrics.Duration.Observe(d.Seconds())
	}
}

// stream dispatches, and retries onto a different replica on failure.
func (p *Proxy) stream(
	ctx context.Context, w http.ResponseWriter, flusher http.Flusher,
	body generateBody, deliveredOut *int,
) error {
	key := PrefixKey(body.Prompt, p.prefixLen)

	// Tokens already forwarded to the client. On a retry these become context
	// for the replacement replica rather than output to regenerate.
	var delivered []uint32
	tried := map[string]bool{}
	var lastErr error

	deadline := float64(0)
	if body.SLOSeconds > 0 {
		deadline = float64(time.Now().Unix()) + body.SLOSeconds
	}

	for attempt := 0; attempt <= p.maxRetries; attempt++ {
		rep, err := p.pickUntried(key, tried)
		if err != nil {
			if lastErr != nil {
				return lastErr
			}
			return err
		}
		tried[rep.ID] = true

		if attempt > 0 {
			p.retries.Add(1)
			if p.metrics != nil {
				p.metrics.Retries.Inc()
			}
			if len(delivered) > 0 {
				p.migrations.Add(1)
				if p.metrics != nil {
					p.metrics.Migrations.Inc()
				}
				p.log.Info("migrating request",
					"request_id", body.RequestID, "to", rep.ID,
					"tokens_already_sent", len(delivered))
			}
		}

		n, err := p.attempt(ctx, w, flusher, rep, body, delivered, deadline)
		p.ring.Release(rep.ID)

		// Report what the client actually received, on every path. This is
		// what the rate limiter refunds against, so undercounting bills the
		// client for tokens it never got and overcounting hands back budget
		// that was really spent.
		if deliveredOut != nil {
			*deliveredOut = len(delivered) + len(n)
		}

		if err == nil {
			return nil
		}
		// The client going away is not a replica failure and must not trigger
		// a retry onto a second replica -- that would double the work for a
		// stream nobody is reading.
		if ctx.Err() != nil {
			return nil
		}
		if isRejection(err) {
			p.rejected.Add(1)
			p.observeOutcome("rejected", 0)
			writeEvent(w, flusher, "rejected", map[string]string{"reason": err.Error()})
			return nil
		}

		delivered = append(delivered, n...)
		lastErr = err
		p.log.Warn("replica failed mid-stream",
			"request_id", body.RequestID, "replica", rep.ID,
			"delivered", len(delivered), "err", err)
	}
	return fmt.Errorf("exhausted %d replicas: %w", len(tried), lastErr)
}

// attempt runs one replica's stream, returning the token ids it delivered.
func (p *Proxy) attempt(
	ctx context.Context, w http.ResponseWriter, flusher http.Flusher,
	rep *Replica, body generateBody, resume []uint32, deadline float64,
) ([]uint32, error) {
	attemptStart := time.Now()
	stub, err := p.pool.Stub(rep.ID)
	if err != nil {
		return nil, err
	}

	req := &GenerateRequest{
		Prompt: body.Prompt,
		Params: &SamplingParams{
			MaxTokens:   body.MaxTokens,
			Temperature: body.Temperature,
			TopK:        body.TopK,
			TopP:        body.TopP,
			IgnoreEos:   body.IgnoreEOS,
		},
		RequestId:    body.RequestID,
		DeadlineUnix: deadline,
		ResumeTokens: resume,
	}

	stream, err := stub.Generate(ctx, req)
	if err != nil {
		return nil, err
	}

	var delivered []uint32
	for {
		chunk, err := stream.Recv()
		if err != nil {
			// io.EOF arrives as a clean end only if the replica sent a finish
			// chunk first; otherwise the stream died and the caller retries.
			if errors.Is(err, context.Canceled) || ctx.Err() != nil {
				return delivered, err
			}
			return delivered, err
		}

		switch payload := chunk.Payload.(type) {
		case *GenerateChunk_Token:
			if len(resume) == 0 && len(delivered) == 0 && p.metrics != nil {
				// Only for a request that started here. On a migration the
				// "first token" the replacement replica produces is not the
				// client's first token -- it already had some -- and counting
				// it would report a suspiciously excellent TTFT precisely when
				// something went wrong.
				p.metrics.TTFT.Observe(time.Since(attemptStart).Seconds())
			}
			delivered = append(delivered, payload.Token.TokenId)
			writeEvent(w, flusher, "token", map[string]any{
				"text":  payload.Token.Text,
				"index": payload.Token.Index,
			})
		case *GenerateChunk_Finish:
			f := payload.Finish
			if f.Reason == Finish_REJECTED {
				return delivered, fmt.Errorf("rejected: %s", f.Message)
			}
			writeEvent(w, flusher, "done", map[string]any{
				"finish_reason": f.Reason.String(),
				"output_tokens": f.OutputTokens,
				"replica":       rep.ID,
			})
			return delivered, nil
		}
	}
}

// pickUntried selects a replica the request has not already failed on.
//
// Retrying onto the same replica that just died is the most common way a
// "retry" achieves nothing, so exhausted replicas are held out and released
// only when the request ends.
func (p *Proxy) pickUntried(key string, tried map[string]bool) (*Replica, error) {
	// One call, with the exclusion applied inside the ring walk. The previous
	// version picked and released in a loop hoping for a different answer,
	// which a deterministic hash never gives.
	return p.ring.PickExcluding(key, tried)
}

func isRejection(err error) bool {
	if err == nil {
		return false
	}
	if s, ok := status.FromError(err); ok && s.Code() == codes.ResourceExhausted {
		return true
	}
	return len(err.Error()) >= 8 && err.Error()[:8] == "rejected"
}

func writeEvent(w http.ResponseWriter, flusher http.Flusher, event string, data any) {
	payload, err := json.Marshal(data)
	if err != nil {
		return
	}
	_, _ = fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event, payload)
	flusher.Flush()
}

// Stats reports router counters.
func (p *Proxy) Stats() map[string]any {
	replicas := p.ring.Snapshot()
	perReplica := make([]map[string]any, 0, len(replicas))
	for _, rep := range replicas {
		perReplica = append(perReplica, map[string]any{
			"id": rep.ID, "addr": rep.Addr, "load": rep.Load, "ready": rep.Ready,
		})
	}
	return map[string]any{
		"requests":   p.requests.Load(),
		"retries":    p.retries.Load(),
		"migrations": p.migrations.Load(),
		"failures":   p.failures.Load(),
		"rejected":   p.rejected.Load(),
		"throttled":  p.throttled.Load(),
		"degraded":   p.degraded.Load(),
		"replicas":   perReplica,
		"ready":      p.ring.ReadyCount(),
		"load_cap":   p.ring.LoadCap(),
		"orphaned":   orphanedOf(p.reconciler),
	}
}

func orphanedOf(rc *Reconciler) []string {
	if rc == nil {
		return nil
	}
	return rc.Orphaned()
}

// clientKey identifies the tenant a request is billed to.
//
// An explicit API key when there is one, falling back to the peer address.
// The fallback is deliberately weak and worth naming: everyone behind one NAT
// shares a bucket, and a client with a pool of addresses gets a bucket each.
// It is the only identity available without authentication, so it is the right
// default and the wrong thing to rely on -- real multi-tenancy needs the key.
func clientKey(r *http.Request) string {
	if key := r.Header.Get("X-API-Key"); key != "" {
		return "key:" + key
	}
	if id := r.Header.Get("X-Client-Id"); id != "" {
		return "client:" + id
	}
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		host = r.RemoteAddr
	}
	return "ip:" + host
}
