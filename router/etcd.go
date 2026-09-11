package router

import (
	"context"
	"fmt"
	"log/slog"
	"strings"
	"time"

	clientv3 "go.etcd.io/etcd/client/v3"
)

// RegistryPrefix is the etcd key space replicas register under.
const RegistryPrefix = "/nanoserve/replicas/"

// EtcdRegistry discovers replicas from etcd, using leases.
//
// A lease is what makes this different from a config file. A replica holds a
// lease and renews it; if the replica dies -- SIGKILL, a partition, a hung
// process -- it stops renewing and etcd deletes the key when the TTL expires.
// Nothing has to notice the death and clean up, which is exactly the step that
// gets skipped in a crash. The router learns within the TTL.
//
// Watch delivers a full snapshot first, then one on every change. A caller
// that fetched-then-subscribed would have to reconcile whatever happened
// between the two calls, and that gap is where a missed deregistration lives.
type EtcdRegistry struct {
	client *clientv3.Client
	log    *slog.Logger
}

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

func (e *EtcdRegistry) Watch(ctx context.Context) (<-chan []Endpoint, error) {
	// Snapshot first, at a known revision, then watch from exactly that
	// revision. Watching from "now" instead would miss any change between the
	// read and the subscribe.
	resp, err := e.client.Get(ctx, RegistryPrefix, clientv3.WithPrefix())
	if err != nil {
		return nil, fmt.Errorf("etcd initial get: %w", err)
	}

	current := map[string]string{}
	for _, kv := range resp.Kvs {
		current[keyToID(string(kv.Key))] = string(kv.Value)
	}

	out := make(chan []Endpoint, 8)
	out <- snapshot(current)

	watcher := e.client.Watch(
		ctx, RegistryPrefix,
		clientv3.WithPrefix(),
		clientv3.WithRev(resp.Header.Revision+1),
	)

	go func() {
		defer close(out)
		for {
			select {
			case <-ctx.Done():
				return
			case wr, ok := <-watcher:
				if !ok {
					return
				}
				if err := wr.Err(); err != nil {
					// etcd being unreachable must not take the router down.
					// The ring keeps whatever membership it already has, which
					// is the documented degraded mode: existing routes keep
					// working, no new registrations arrive.
					e.log.Warn("etcd watch error; serving last known membership",
						"err", err)
					continue
				}
				for _, ev := range wr.Events {
					id := keyToID(string(ev.Kv.Key))
					switch ev.Type {
					case clientv3.EventTypePut:
						current[id] = string(ev.Kv.Value)
						e.log.Info("replica registered", "replica", id,
							"addr", string(ev.Kv.Value))
					case clientv3.EventTypeDelete:
						// A delete is usually a lease expiring, i.e. a replica
						// that died without deregistering.
						delete(current, id)
						e.log.Info("replica gone", "replica", id)
					}
				}
				select {
				case out <- snapshot(current):
				case <-ctx.Done():
					return
				}
			}
		}
	}()

	return out, nil
}

func (e *EtcdRegistry) Close() error {
	return e.client.Close()
}

// Register announces a replica under a lease and keeps it alive.
//
// Returns a stop function that revokes the lease, which is the graceful path:
// the key disappears immediately rather than after the TTL, so a planned
// shutdown removes the replica from rotation in milliseconds instead of
// seconds.
func Register(
	ctx context.Context, endpoints []string, id, addr string, ttlSeconds int64,
	log *slog.Logger,
) (func(), error) {
	client, err := clientv3.New(clientv3.Config{
		Endpoints:   endpoints,
		DialTimeout: 5 * time.Second,
	})
	if err != nil {
		return nil, fmt.Errorf("etcd connect: %w", err)
	}

	lease, err := client.Grant(ctx, ttlSeconds)
	if err != nil {
		_ = client.Close()
		return nil, fmt.Errorf("grant lease: %w", err)
	}

	key := RegistryPrefix + id
	if _, err := client.Put(ctx, key, addr, clientv3.WithLease(lease.ID)); err != nil {
		_ = client.Close()
		return nil, fmt.Errorf("register %s: %w", id, err)
	}

	keepAlive, err := client.KeepAlive(ctx, lease.ID)
	if err != nil {
		_ = client.Close()
		return nil, fmt.Errorf("keepalive %s: %w", id, err)
	}

	go func() {
		for range keepAlive {
			// Draining the channel is the renewal. If this goroutine stops --
			// because the process died -- renewals stop and the lease expires,
			// which is precisely the detection mechanism.
		}
		log.Warn("lease keepalive ended", "replica", id)
	}()

	log.Info("registered", "replica", id, "addr", addr, "ttl_seconds", ttlSeconds)

	return func() {
		revokeCtx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()
		if _, err := client.Revoke(revokeCtx, lease.ID); err != nil {
			log.Warn("lease revoke failed; will expire on TTL", "err", err)
		}
		_ = client.Close()
	}, nil
}

func keyToID(key string) string {
	return strings.TrimPrefix(key, RegistryPrefix)
}

func snapshot(current map[string]string) []Endpoint {
	out := make([]Endpoint, 0, len(current))
	for id, addr := range current {
		out = append(out, Endpoint{ID: id, Addr: addr})
	}
	return out
}
