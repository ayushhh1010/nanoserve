package router

import (
	"time"

	"github.com/prometheus/client_golang/prometheus"
)

// Router metrics.
//
// Naming follows the Prometheus conventions the serving ecosystem already
// uses -- `_total` on counters, base units (seconds, never milliseconds), and
// a subsystem prefix -- so these sit beside the engine's vLLM-style metrics in
// one dashboard without a translation layer.
//
// Cardinality is the thing to get right, and it is easy to get wrong in a
// router. Labelling by replica is safe: replicas are few and long-lived.
// Labelling by client, request id, or prompt would be unbounded, and an
// unbounded label is not a slow dashboard -- it is a Prometheus that falls
// over and takes the observability of an incident with it, at the moment the
// incident is happening. So client identity appears in logs, never in labels.

// Buckets for end-to-end request latency. Generation takes seconds, so the
// default client buckets (which top out at 10s) would pile everything into
// +Inf and make the tail unreadable.
var requestDurationBuckets = []float64{
	0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64,
}

// TTFT deserves its own, tighter buckets. It is the number a user feels, it
// lives in the sub-second range where the request-duration buckets have almost
// no resolution, and a p99 measured through the wrong buckets is a guess.
var ttftBuckets = []float64{
	0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10,
}

// Metrics is the router's Prometheus surface.
type Metrics struct {
	Requests   *prometheus.CounterVec
	Retries    prometheus.Counter
	Migrations prometheus.Counter
	Throttled  *prometheus.CounterVec
	Degraded   prometheus.Counter

	Duration prometheus.Histogram
	TTFT     prometheus.Histogram

	Replicas    *prometheus.GaugeVec
	ReplicaLoad *prometheus.GaugeVec
}

func NewMetrics(reg prometheus.Registerer) *Metrics {
	m := &Metrics{
		Requests: prometheus.NewCounterVec(prometheus.CounterOpts{
			Namespace: "nanoserve", Subsystem: "router", Name: "requests_total",
			Help: "Requests by terminal outcome as the client saw it.",
		}, []string{"outcome"}),

		Retries: prometheus.NewCounter(prometheus.CounterOpts{
			Namespace: "nanoserve", Subsystem: "router", Name: "retries_total",
			Help: "Dispatch attempts onto a replacement replica.",
		}),

		Migrations: prometheus.NewCounter(prometheus.CounterOpts{
			Namespace: "nanoserve", Subsystem: "router", Name: "migrations_total",
			Help: "Retries that carried already-delivered tokens as resume context. " +
				"Distinct from retries: a migration means a client was mid-stream " +
				"when its replica died, which is the expensive case.",
		}),

		Throttled: prometheus.NewCounterVec(prometheus.CounterOpts{
			Namespace: "nanoserve", Subsystem: "router", Name: "throttled_total",
			Help: "Requests refused by the rate limiter.",
		}, []string{"reason"}),

		Degraded: prometheus.NewCounter(prometheus.CounterOpts{
			Namespace: "nanoserve", Subsystem: "router", Name: "ratelimit_degraded_total",
			Help: "Admissions granted because the limiter was unreachable. " +
				"Non-zero means limits are not being enforced -- an outage that " +
				"is otherwise invisible, because everything succeeds.",
		}),

		Duration: prometheus.NewHistogram(prometheus.HistogramOpts{
			Namespace: "nanoserve", Subsystem: "router",
			Name: "request_duration_seconds", Buckets: requestDurationBuckets,
			Help: "End-to-end, router-observed, successful requests only. " +
				"Mixing failures in makes a fast failure look like fast service.",
		}),

		TTFT: prometheus.NewHistogram(prometheus.HistogramOpts{
			Namespace: "nanoserve", Subsystem: "router",
			Name: "time_to_first_token_seconds", Buckets: ttftBuckets,
			Help: "Router-observed time to the first token forwarded to a client.",
		}),

		Replicas: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Namespace: "nanoserve", Subsystem: "router", Name: "replicas",
			Help: "Replica count by state. 'orphaned' are replicas the registry " +
				"dropped but health still vouches for -- a degraded control " +
				"plane serving on retained membership.",
		}, []string{"state"}),

		ReplicaLoad: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Namespace: "nanoserve", Subsystem: "router", Name: "replica_inflight",
			Help: "In-flight requests per replica, as the ring accounts for them. " +
				"The spread across replicas is what bounded-load routing is for.",
		}, []string{"replica"}),
	}

	reg.MustRegister(
		m.Requests, m.Retries, m.Migrations, m.Throttled, m.Degraded,
		m.Duration, m.TTFT, m.Replicas, m.ReplicaLoad,
	)
	return m
}

// ObserveRing publishes ring state. Called on a timer rather than on every
// routing decision: these are gauges describing the fleet, and recomputing
// them inside the hot path would put a metrics write under the ring lock.
func (m *Metrics) ObserveRing(ring *Ring, rc *Reconciler) {
	snapshot := ring.Snapshot()
	ready := 0
	for _, rep := range snapshot {
		if rep.Ready {
			ready++
		}
		m.ReplicaLoad.WithLabelValues(rep.ID).Set(float64(rep.Load))
	}
	m.Replicas.WithLabelValues("registered").Set(float64(len(snapshot)))
	m.Replicas.WithLabelValues("ready").Set(float64(ready))
	orphaned := 0
	if rc != nil {
		orphaned = len(rc.Orphaned())
	}
	m.Replicas.WithLabelValues("orphaned").Set(float64(orphaned))
}

// RunRingObserver refreshes ring gauges until ctx ends.
func (m *Metrics) RunRingObserver(done <-chan struct{}, ring *Ring, rc *Reconciler, every time.Duration) {
	ticker := time.NewTicker(every)
	defer ticker.Stop()
	for {
		select {
		case <-done:
			return
		case <-ticker.C:
			m.ObserveRing(ring, rc)
		}
	}
}
