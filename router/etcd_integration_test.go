package router

import (
	"context"
	"io"
	"log/slog"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"

	clientv3 "go.etcd.io/etcd/client/v3"
)

// These run against a real etcd, because the things worth testing here are
// etcd's semantics, not ours. A fake registry would happily confirm whatever
// the fake was written to believe -- and the two bugs this file exists to
// catch (a keepalive on a dead lease returning success, a watch that never
// recovers) are both cases where the real server does something a reasonable
// fake would not.
//
//	docker run -d --name nanoserve-etcd -p 2379:2379 registry.k8s.io/etcd:3.7.1-0 \
//	  etcd --name n1 --listen-client-urls http://0.0.0.0:2379 \
//	       --advertise-client-urls http://127.0.0.1:2379 --data-dir /tmp/etcd
//	NANOSERVE_ETCD=127.0.0.1:2379 go test ./... -run Etcd -v
func etcdEndpoints(t *testing.T) []string {
	t.Helper()
	v := os.Getenv("NANOSERVE_ETCD")
	if v == "" {
		t.Skip("set NANOSERVE_ETCD=host:port to run etcd integration tests")
	}
	return strings.Split(v, ",")
}

func quietLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(io.Discard, nil))
}

// testClient dials etcd and cleans up the registry prefix around the test, so
// a failed run cannot poison the next one.
func testClient(t *testing.T) *clientv3.Client {
	t.Helper()
	cli, err := clientv3.New(clientv3.Config{
		Endpoints:   etcdEndpoints(t),
		DialTimeout: 5 * time.Second,
	})
	if err != nil {
		t.Fatalf("dial etcd: %v", err)
	}
	clean := func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_, _ = cli.Delete(ctx, RegistryPrefix, clientv3.WithPrefix())
	}
	clean()
	t.Cleanup(func() {
		clean()
		_ = cli.Close()
	})
	return cli
}

// announce registers a replica the way engine/discovery.py does: a lease, a
// key under it, and renewals for as long as the replica is alive. The
// returned cancel stops renewing *without* revoking -- which is what a crash
// looks like from etcd's side.
func announce(
	t *testing.T, cli *clientv3.Client, id, addr string, ttl int64,
) context.CancelFunc {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())

	lease, err := cli.Grant(ctx, ttl)
	if err != nil {
		cancel()
		t.Fatalf("grant: %v", err)
	}
	if _, err := cli.Put(ctx, RegistryPrefix+id, addr, clientv3.WithLease(lease.ID)); err != nil {
		cancel()
		t.Fatalf("put: %v", err)
	}
	ka, err := cli.KeepAlive(ctx, lease.ID)
	if err != nil {
		cancel()
		t.Fatalf("keepalive: %v", err)
	}
	go func() {
		for range ka {
		}
	}()
	return cancel
}

// awaitMembership drains the channel until the set of replica IDs matches, or
// the deadline passes. Reading one event and asserting on it would be flaky:
// registration and health are separate events and may arrive in either order.
func awaitMembership(
	t *testing.T, events <-chan []Endpoint, want []string, within time.Duration,
) {
	t.Helper()
	deadline := time.After(within)
	var last []string
	for {
		select {
		case eps, ok := <-events:
			if !ok {
				t.Fatalf("membership channel closed; want %v, last saw %v", want, last)
			}
			last = ids(eps)
			if equalSets(last, want) {
				return
			}
		case <-deadline:
			t.Fatalf("timed out waiting for membership %v; last saw %v", want, last)
		}
	}
}

func ids(eps []Endpoint) []string {
	out := make([]string, 0, len(eps))
	for _, e := range eps {
		out = append(out, e.ID)
	}
	return out
}

func equalSets(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	seen := make(map[string]int, len(a))
	for _, x := range a {
		seen[x]++
	}
	for _, y := range b {
		seen[y]--
		if seen[y] < 0 {
			return false
		}
	}
	return true
}

// TestEtcdWatchSeesRegistration is the baseline: a replica that registers
// after the watch started must show up without the router restarting.
func TestEtcdWatchSeesRegistration(t *testing.T) {
	cli := testClient(t)
	reg, err := NewEtcdRegistry(etcdEndpoints(t), quietLogger())
	if err != nil {
		t.Fatalf("registry: %v", err)
	}
	t.Cleanup(func() { _ = reg.Close() })

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	events, err := reg.Watch(ctx)
	if err != nil {
		t.Fatalf("watch: %v", err)
	}
	awaitMembership(t, events, nil, 5*time.Second) // empty registry first

	stop := announce(t, cli, "replica-a", "127.0.0.1:9101", 10)
	defer stop()
	awaitMembership(t, events, []string{"replica-a"}, 5*time.Second)

	stop2 := announce(t, cli, "replica-b", "127.0.0.1:9102", 10)
	defer stop2()
	awaitMembership(t, events, []string{"replica-a", "replica-b"}, 5*time.Second)
}

// TestEtcdLeaseExpiryRemovesDeadReplica is the reason discovery uses leases at
// all. The replica does not deregister -- it simply stops renewing, the way a
// SIGKILLed process does -- and must leave the ring anyway.
func TestEtcdLeaseExpiryRemovesDeadReplica(t *testing.T) {
	cli := testClient(t)
	reg, err := NewEtcdRegistry(etcdEndpoints(t), quietLogger())
	if err != nil {
		t.Fatalf("registry: %v", err)
	}
	t.Cleanup(func() { _ = reg.Close() })

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	alive := announce(t, cli, "replica-alive", "127.0.0.1:9101", 10)
	defer alive()
	doomed := announce(t, cli, "replica-doomed", "127.0.0.1:9102", 2)

	events, err := reg.Watch(ctx)
	if err != nil {
		t.Fatalf("watch: %v", err)
	}
	awaitMembership(t, events, []string{"replica-alive", "replica-doomed"}, 5*time.Second)

	// Stop renewing without revoking: a crash, not a shutdown.
	doomed()

	// Generous window: expiry is bounded by the TTL plus however long etcd
	// takes to run its lease sweep, which is not instantaneous.
	awaitMembership(t, events, []string{"replica-alive"}, 20*time.Second)
}

// TestEtcdRevokeIsFasterThanTTL covers the graceful path. A planned shutdown
// revokes, and the point of revoking is that rotation updates in milliseconds
// rather than after the TTL -- that difference is what makes a rolling deploy
// drop nothing.
func TestEtcdRevokeIsFasterThanTTL(t *testing.T) {
	cli := testClient(t)
	reg, err := NewEtcdRegistry(etcdEndpoints(t), quietLogger())
	if err != nil {
		t.Fatalf("registry: %v", err)
	}
	t.Cleanup(func() { _ = reg.Close() })

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	const ttl = 30 // far longer than the test may take, so only revoke can pass it
	lease, err := cli.Grant(ctx, ttl)
	if err != nil {
		t.Fatalf("grant: %v", err)
	}
	if _, err := cli.Put(ctx, RegistryPrefix+"replica-bye", "127.0.0.1:9101",
		clientv3.WithLease(lease.ID)); err != nil {
		t.Fatalf("put: %v", err)
	}

	events, err := reg.Watch(ctx)
	if err != nil {
		t.Fatalf("watch: %v", err)
	}
	awaitMembership(t, events, []string{"replica-bye"}, 5*time.Second)

	start := time.Now()
	if _, err := cli.Revoke(ctx, lease.ID); err != nil {
		t.Fatalf("revoke: %v", err)
	}
	awaitMembership(t, events, nil, 5*time.Second)

	if elapsed := time.Since(start); elapsed > 2*time.Second {
		t.Fatalf("revoke took %v to clear rotation; TTL was %ds -- revoke is "+
			"supposed to be the fast path", elapsed, ttl)
	}
}

// TestEtcdWatchSurvivesRestart asserts the property that actually matters:
// after etcd goes away and comes back, the router still learns about new
// replicas. It deliberately does NOT assert *which layer* recovered.
//
// An earlier version of this test required Reestablished() > 0, on the
// assumption that killing etcd would kill the watch and force our supervisor
// to rebuild it. Measured, that is false three times over: a container
// restart, and even destroying and recreating the container with a fresh data
// directory, are both absorbed by clientv3's own retry -- the watch channel
// never closes and the supervisor never runs. The supervisor earns its place
// on the failures clientv3 gives up on instead (see
// TestEtcdConsumeReturnsOnCompaction), not on a process restart.
//
// Keeping the stronger assertion would have been worse than useless: it fails
// on correct code, for a reason that has nothing to do with the behaviour
// under test.
func TestEtcdWatchSurvivesRestart(t *testing.T) {
	container := os.Getenv("NANOSERVE_ETCD_CONTAINER")
	if container == "" {
		t.Skip("set NANOSERVE_ETCD_CONTAINER=<name> to run the restart test")
	}
	cli := testClient(t)
	reg, err := NewEtcdRegistry(etcdEndpoints(t), quietLogger())
	if err != nil {
		t.Fatalf("registry: %v", err)
	}
	t.Cleanup(func() { _ = reg.Close() })

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	stopA := announce(t, cli, "replica-a", "127.0.0.1:9101", 60)
	defer stopA()

	events, err := reg.Watch(ctx)
	if err != nil {
		t.Fatalf("watch: %v", err)
	}
	awaitMembership(t, events, []string{"replica-a"}, 5*time.Second)

	// The control plane goes away underneath a running router.
	if out, err := exec.Command("docker", "restart", container).CombinedOutput(); err != nil {
		t.Fatalf("docker restart %s: %v: %s", container, err, out)
	}

	waitFor(t, 60*time.Second, "etcd to accept writes again", func() bool {
		c, cancel := context.WithTimeout(ctx, 2*time.Second)
		defer cancel()
		_, err := cli.Get(c, RegistryPrefix)
		return err == nil
	})

	// A registration made after the restart is the proof: the router can only
	// see it if discovery is still live.
	stopB := announce(t, cli, "replica-b", "127.0.0.1:9102", 60)
	defer stopB()
	awaitMembership(t, events, []string{"replica-a", "replica-b"}, 60*time.Second)

	t.Logf("discovery survived an etcd restart; supervisor rebuilds: %d "+
		"(0 means clientv3 absorbed it, which is the expected path)",
		reg.Reestablished())
}

// TestEtcdConsumeReturnsOnCompaction covers the other way a watcher dies. It
// drives consume() directly with a revision that has been compacted away,
// because reaching that state through the public Watch is not possible by
// design -- the loop never falls behind. The requirement is that consume
// *returns* so the supervisor can re-snapshot, rather than blocking forever
// on a stream that will never deliver anything again.
func TestEtcdConsumeReturnsOnCompaction(t *testing.T) {
	cli := testClient(t)
	reg, err := NewEtcdRegistry(etcdEndpoints(t), quietLogger())
	if err != nil {
		t.Fatalf("registry: %v", err)
	}
	t.Cleanup(func() { _ = reg.Close() })

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	start, err := cli.Get(ctx, RegistryPrefix)
	if err != nil {
		t.Fatalf("read revision: %v", err)
	}
	staleRev := start.Header.Revision

	for i := 0; i < 200; i++ {
		if _, err := cli.Put(ctx, "/nanoserve/churn", "x"); err != nil {
			t.Fatalf("churn: %v", err)
		}
	}
	head, err := cli.Get(ctx, "/nanoserve/churn")
	if err != nil {
		t.Fatalf("read head: %v", err)
	}
	if _, err := cli.Compact(ctx, head.Header.Revision); err != nil {
		t.Fatalf("compact: %v", err)
	}

	done := make(chan struct{})
	go func() {
		defer close(done)
		reg.consume(ctx, map[string]string{}, staleRev, make(chan []Endpoint, 8))
	}()

	select {
	case <-done:
	case <-time.After(15 * time.Second):
		t.Fatal("consume did not return on a compacted revision; the supervisor " +
			"can never re-snapshot and discovery is dead until restart")
	}
}

// TestEtcdReconcilerAppliesMembershipToRing checks the seam between discovery
// and routing. Membership is not routing: a replica that appears in etcd must
// land on the ring, and one whose lease expired must leave it -- but a newly
// discovered replica must NOT be routable until health says so, which is the
// other half of a bug that cost every request in the first end-to-end run.
func TestEtcdReconcilerAppliesMembershipToRing(t *testing.T) {
	cli := testClient(t)
	reg, err := NewEtcdRegistry(etcdEndpoints(t), quietLogger())
	if err != nil {
		t.Fatalf("registry: %v", err)
	}
	t.Cleanup(func() { _ = reg.Close() })

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	events, err := reg.Watch(ctx)
	if err != nil {
		t.Fatalf("watch: %v", err)
	}

	ring := NewRing(DefaultVirtualNodes, DefaultEpsilon)
	go NewReconciler(ring, quietLogger()).Run(ctx, events)

	stop := announce(t, cli, "replica-a", "127.0.0.1:9101", 30)
	defer stop()

	waitFor(t, 10*time.Second, "replica on ring", func() bool {
		return len(ring.Snapshot()) == 1
	})
	if n := ring.ReadyCount(); n != 0 {
		t.Fatalf("newly discovered replica is ready=%d before any health check; "+
			"a registry entry means a replica exists, not that it can serve", n)
	}

	ring.SetReady("replica-a", true)
	if n := ring.ReadyCount(); n != 1 {
		t.Fatalf("ReadyCount = %d after health passed, want 1", n)
	}
}

func waitFor(t *testing.T, within time.Duration, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(within)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(50 * time.Millisecond)
	}
	t.Fatalf("timed out after %v waiting for %s", within, what)
}
