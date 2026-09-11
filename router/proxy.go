package router

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
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

	// Counters for /metrics. Atomic rather than mutex-guarded: they are
	// touched on every request and never read in the same critical section as
	// anything else.
	requests   atomic.Int64
	retries    atomic.Int64
	migrations atomic.Int64
	failures   atomic.Int64
	rejected   atomic.Int64
}

func NewProxy(ring *Ring, pool *ClientPool, prefixLen, maxRetries int, log *slog.Logger) *Proxy {
	return &Proxy{
		ring: ring, pool: pool, log: log,
		prefixLen: prefixLen, maxRetries: maxRetries,
	}
}

type generateBody struct {
	Prompt      string   `json:"prompt"`
	MaxTokens   uint32   `json:"max_tokens"`
	Temperature float32  `json:"temperature"`
	TopK        uint32   `json:"top_k"`
	TopP        float32  `json:"top_p"`
	IgnoreEOS   bool     `json:"ignore_eos"`
	SLOSeconds  float64  `json:"slo_seconds"`
	RequestID   string   `json:"request_id"`
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

	p.requests.Add(1)
	if err := p.stream(r.Context(), w, flusher, body); err != nil {
		p.failures.Add(1)
		p.log.Error("request failed", "request_id", body.RequestID, "err", err)
		writeEvent(w, flusher, "error", map[string]string{"error": err.Error()})
	}
}

// stream dispatches, and retries onto a different replica on failure.
func (p *Proxy) stream(
	ctx context.Context, w http.ResponseWriter, flusher http.Flusher, body generateBody,
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
			if len(delivered) > 0 {
				p.migrations.Add(1)
				p.log.Info("migrating request",
					"request_id", body.RequestID, "to", rep.ID,
					"tokens_already_sent", len(delivered))
			}
		}

		n, err := p.attempt(ctx, w, flusher, rep, body, delivered, deadline)
		p.ring.Release(rep.ID)

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
		RequestId:     body.RequestID,
		DeadlineUnix:  deadline,
		ResumeTokens:  resume,
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
	for i := 0; i < p.ring.Len()+1; i++ {
		rep, err := p.ring.Pick(key)
		if err != nil {
			return nil, err
		}
		if !tried[rep.ID] {
			return rep, nil
		}
		// Return the reservation before looking again, or the load accounting
		// drifts upward on every retry.
		p.ring.Release(rep.ID)
	}
	return nil, ErrNoReplicas
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
		"replicas":   perReplica,
		"ready":      p.ring.ReadyCount(),
		"load_cap":   p.ring.LoadCap(),
	}
}
