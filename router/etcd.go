package router

import (
	"context"
	"fmt"
	"log/slog"
	"math/rand/v2"
	"strings"
	"sync/atomic"
	"time"

	clientv3 "go.etcd.io/etcd/client/v3"
)

// RegistryPrefix is the etcd key space replicas register under.
//
// Must match REGISTRY_PREFIX in engine/discovery.py. Replicas write here;
// the router only ever reads. That split is deliberate: the thing that knows
// whether a replica is alive is the replica's own process, and a router that
// also wrote membership would be guessing.
const RegistryPrefix = "/nanoserve/replicas/"

// EtcdRegistry discovers replicas from etcd, using leases.
//
// A lease is what makes this different from a config file. A replica holds a
// lease and renews it; if it dies -- SIGKILL, a partition, a hung process --
// renewals stop and etcd deletes the key when the TTL expires. Nothing has to
// notice the death and clean up, which is exactly the step that gets skipped
// in a crash.
//
// Watch delivers a full snapshot first, then one on every change. A caller
// that fetched-then-subscribed would have to reconcile whatever happened
// between the two calls, and that gap is where a missed deregistration lives.
type EtcdRegistry struct {
	client *clientv3.Client
	log    *slog.Logger

	// reestablished counts watchers rebuilt after one died. Exported through
	// Reestablished() only so the recovery test can prove it actually
	// exercised recovery: a test that asserts "membership still updates"
	// passes just as happily when the watcher never broke in the first place,
	// which is how a regression test quietly stops testing its regression.
	reestablished atomic.Int64
}

// Reestablished reports how many times the watch was rebuilt after failure.
func (e *EtcdRegistry) Reestablished() int64 { return e.reestablished.Load() }

func NewEtcdRegistry(endpoints []string, log *slog.Logger) (*EtcdRegistry, error) {
	client, err := clientv3.New(clientv3.Config{
		Endpoints:   endpoints,
		DialTimeout: 5 * time.Second,
	})
	if err != nil {
		return nil, fmt.Errorf("etcd connect: %w", err)
	}
	return &EtcdRegistry{client: client, log: log}, nil
}

// Watch streams membership until ctx is cancelled.
//
// The loop outlives any single etcd watcher, and that is the entire point.
// clientv3 retries what it can, but it closes the channel on anything it
// cannot handle -- a compacted revision, a stream the server cancelled, a
// cluster that went away for long enough. A watch loop that simply returns
// there stops reconciling membership *permanently*, including after etcd
// comes back: the router goes on routing to replicas whose leases expired
// while it was not looking. Losing etcd should degrade discovery for as long
// as etcd is down, and no longer.
//
// Recovery re-reads a snapshot rather than resuming from the last revision.
// The old revision may be exactly what got compacted, and a snapshot also
// reconciles every change missed while the watch was broken, which resuming
// cannot do.
func (e *EtcdRegistry) Watch(ctx context.Context) (<-chan []Endpoint, error) {
	// The first read is synchronous so a misconfigured or unreachable etcd
	// fails at startup, where the operator sees it, instead of coming up
	// healthy with an empty ring and refusing every request.
	members, rev, err := e.snapshot(ctx)
	if err != nil {
		return nil, err
	}

	out := make(chan []Endpoint, 8)
	out <- endpointsOf(members)

	go func() {
		defer close(out)
		var bo backoff
		for {
			// Blocks until this watcher dies, publishing changes as they land.
			e.consume(ctx, members, rev, out)
			if ctx.Err() != nil {
				return
			}
			e.log.Warn("etcd watch ended; re-establishing")
			if !sleepCtx(ctx, bo.next()) {
				return
			}
			next, nextRev, err := e.snapshot(ctx)
			if err != nil {
				// Degraded, not dead. The ring keeps the membership it has,
				// so in-flight and new requests still route; only *changes*
				// are invisible until etcd answers again.
				e.log.Warn("etcd unreachable; serving last known membership",
					"err", err)
				continue
			}
			bo.reset()
			members, rev = next, nextRev
			e.reestablished.Add(1)
			e.log.Info("etcd watch re-established", "replicas", len(members))
			if !send(ctx, out, endpointsOf(members)) {
				return
			}
		}
	}()

	return out, nil
}

// consume runs one watcher to exhaustion, mutating members as events arrive.
func (e *EtcdRegistry) consume(
	ctx context.Context, members map[string]string, rev int64, out chan []Endpoint,
) {
	watcher := e.client.Watch(
		ctx, RegistryPrefix,
		clientv3.WithPrefix(),
		// rev+1: start at the first revision *after* the snapshot, so no
		// event is either missed or replayed.
		clientv3.WithRev(rev+1),
	)

	for {
		select {
		case <-ctx.Done():
			return
		case wr, ok := <-watcher:
			if !ok {
				return
			}
			if err := wr.Err(); err != nil {
				// Compaction, or a stream the server cancelled. The channel
				// closes right after this, so returning hands control to the
				// supervisor, which re-snapshots.
				e.log.Warn("etcd watch error", "err", err)
				return
			}
			for _, ev := range wr.Events {
				id := keyToID(string(ev.Kv.Key))
				switch ev.Type {
				case clientv3.EventTypePut:
					members[id] = string(ev.Kv.Value)
					e.log.Info("replica registered", "replica", id,
						"addr", string(ev.Kv.Value))
				case clientv3.EventTypeDelete:
					// Usually a lease expiring: a replica that died without
					// getting to deregister itself.
					delete(members, id)
					e.log.Info("replica gone", "replica", id)
				}
			}
			if !send(ctx, out, endpointsOf(members)) {
				return
			}
		}
	}
}

// snapshot reads the whole registry and the revision it was consistent at.
func (e *EtcdRegistry) snapshot(ctx context.Context) (map[string]string, int64, error) {
	// Per-call timeout: without one, a Get against a partitioned etcd blocks
	// until ctx dies, which turns the retry loop into a single silent hang.
	cctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	resp, err := e.client.Get(cctx, RegistryPrefix, clientv3.WithPrefix())
	if err != nil {
		return nil, 0, fmt.Errorf("etcd get %s: %w", RegistryPrefix, err)
	}
	members := make(map[string]string, len(resp.Kvs))
	for _, kv := range resp.Kvs {
		members[keyToID(string(kv.Key))] = string(kv.Value)
	}
	return members, resp.Header.Revision, nil
}

func (e *EtcdRegistry) Close() error {
	return e.client.Close()
}

// send publishes a snapshot, discarding any queued one it supersedes.
//
// Every message is the complete membership set, never a delta, so a newer
// snapshot makes an older queued one simply wrong -- there is nothing to lose
// by dropping it. Blocking instead would stall the watch loop behind a slow
// consumer, and a stalled watch loop is how events get missed.
func send(ctx context.Context, out chan []Endpoint, eps []Endpoint) bool {
	for {
		select {
		case out <- eps:
			return true
		case <-ctx.Done():
			return false
		default:
		}
		select {
		case <-out: // drop the stale snapshot, then retry the send
		case <-ctx.Done():
			return false
		}
	}
}

// backoff produces retry delays that grow, then stop growing, with jitter.
type backoff struct{ attempt int }

const (
	backoffBase = 100 * time.Millisecond
	backoffMax  = 5 * time.Second
)

func (b *backoff) next() time.Duration {
	d := backoffBase << min(b.attempt, 8)
	if d > backoffMax || d <= 0 {
		d = backoffMax
	}
	b.attempt++
	// Equal jitter: half fixed, half random. Undithered exponential backoff
	// synchronises -- every client lost etcd at the same instant, so they all
	// retry at the same instant, and that retry storm is what stops etcd
	// coming back.
	return d/2 + time.Duration(rand.Int64N(int64(d/2)+1))
}

func (b *backoff) reset() { b.attempt = 0 }

// sleepCtx waits for d. Reports false if ctx ended first.
func sleepCtx(ctx context.Context, d time.Duration) bool {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return false
	case <-t.C:
		return true
	}
}

func keyToID(key string) string {
	return strings.TrimPrefix(key, RegistryPrefix)
}

func endpointsOf(members map[string]string) []Endpoint {
	out := make([]Endpoint, 0, len(members))
	for id, addr := range members {
		out = append(out, Endpoint{ID: id, Addr: addr})
	}
	return out
}
