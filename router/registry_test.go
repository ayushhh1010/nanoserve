package router

import (
	"testing"
	"time"
)

// TestReconcilerRetainsHealthyDeregisteredReplica is the regression test for
// the chaos suite's etcd_down failure.
//
// Stopping etcd made it broadcast lease-expiry deletes for every replica on
// the way down -- the leases genuinely had lapsed, because the replicas could
// not renew against an etcd that was shutting down. The router believed it and
// emptied the ring, returning "no replicas available" for 26,437 requests
// while two healthy replicas answered every health check put to them. A
// control-plane outage became a total outage.
func TestReconcilerRetainsHealthyDeregisteredReplica(t *testing.T) {
	ring := NewRing(DefaultVirtualNodes, DefaultEpsilon)
	rc := NewReconciler(ring, quietLogger())

	rc.apply([]Endpoint{{ID: "a", Addr: "a:1"}, {ID: "b", Addr: "b:1"}})
	ring.SetReady("a", true)
	ring.SetReady("b", true)
	if ring.ReadyCount() != 2 {
		t.Fatalf("setup: ReadyCount = %d, want 2", ring.ReadyCount())
	}

	// Discovery reports everything gone, while both replicas are healthy.
	rc.apply(nil)

	if got := ring.ReadyCount(); got != 2 {
		t.Fatalf("ReadyCount = %d after discovery reported an empty registry; "+
			"both replicas are passing health checks, and believing discovery "+
			"here turns an etcd outage into a total outage", got)
	}
	if len(rc.Orphaned()) != 2 {
		t.Fatalf("Orphaned() = %v, want both replicas flagged; running on "+
			"retained membership is a degraded state and must be visible",
			rc.Orphaned())
	}
}

// TestReconcilerReapsDeregisteredReplicaOnceUnhealthy is the other half. The
// retention must be temporary, or a genuinely dead replica is kept forever.
func TestReconcilerReapsDeregisteredReplicaOnceUnhealthy(t *testing.T) {
	ring := NewRing(DefaultVirtualNodes, DefaultEpsilon)
	rc := NewReconciler(ring, quietLogger())

	rc.apply([]Endpoint{{ID: "a", Addr: "a:1"}, {ID: "b", Addr: "b:1"}})
	ring.SetReady("a", true)
	ring.SetReady("b", true)

	// "b" dies: its lease expires and, a health interval later, it stops
	// answering. This is the SIGKILL ordering -- membership first, health
	// second -- and the reason reaping runs on a timer rather than only on
	// membership events: a dead replica produces exactly one event, and at
	// that instant health has not failed yet.
	rc.apply([]Endpoint{{ID: "a", Addr: "a:1"}})
	if ring.Len() != 2 {
		t.Fatalf("ring.Len() = %d immediately after deregistration; the "+
			"retention is what gives health time to confirm", ring.Len())
	}

	ring.SetReady("b", false)
	rc.reap()

	if ring.Len() != 1 {
		t.Fatalf("ring.Len() = %d after the deregistered replica also failed "+
			"health; retention must be temporary or corpses accumulate",
			ring.Len())
	}
	if len(rc.Orphaned()) != 0 {
		t.Fatalf("Orphaned() = %v after reaping", rc.Orphaned())
	}
}

// TestReconcilerRemovesUnhealthyDeregisteredImmediately: when both signals
// agree the replica is gone, there is nothing to wait for.
func TestReconcilerRemovesUnhealthyDeregisteredImmediately(t *testing.T) {
	ring := NewRing(DefaultVirtualNodes, DefaultEpsilon)
	rc := NewReconciler(ring, quietLogger())

	rc.apply([]Endpoint{{ID: "a", Addr: "a:1"}, {ID: "b", Addr: "b:1"}})
	ring.SetReady("a", true)
	// "b" never became ready -- it was registered and then died during startup.

	rc.apply([]Endpoint{{ID: "a", Addr: "a:1"}})
	if ring.Len() != 1 {
		t.Fatalf("ring.Len() = %d; a replica that is both deregistered and "+
			"unhealthy has no evidence in its favour", ring.Len())
	}
}

// TestReconcilerReadmitsReplicaThatComesBack covers recovery: when etcd
// returns and the replica re-registers, it must stop being an orphan and must
// not lose its readiness in the process.
func TestReconcilerReadmitsReplicaThatComesBack(t *testing.T) {
	ring := NewRing(DefaultVirtualNodes, DefaultEpsilon)
	rc := NewReconciler(ring, quietLogger())

	rc.apply([]Endpoint{{ID: "a", Addr: "a:1"}})
	ring.SetReady("a", true)
	rc.apply(nil)
	if len(rc.Orphaned()) != 1 {
		t.Fatalf("Orphaned() = %v, want [a]", rc.Orphaned())
	}

	rc.apply([]Endpoint{{ID: "a", Addr: "a:1"}})
	if len(rc.Orphaned()) != 0 {
		t.Fatalf("Orphaned() = %v after re-registration", rc.Orphaned())
	}
	if ring.ReadyCount() != 1 {
		t.Fatalf("ReadyCount = %d after re-registration; re-adding an existing "+
			"replica must not reset its health and blip it out of rotation",
			ring.ReadyCount())
	}
	_ = time.Second
}
