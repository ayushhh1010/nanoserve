package router

import (
	"context"
	"fmt"
	"sync"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/keepalive"
)

// ClientPool holds one gRPC connection per replica.
//
// One connection, not one per request. HTTP/2 multiplexes every stream for a
// replica over a single connection, so the router holds N connections for N
// replicas rather than one per in-flight generation -- which is the entire
// reason a proxy like this can hold thousands of concurrent streams on a
// machine that could not hold thousands of sockets.
type ClientPool struct {
	mu    sync.RWMutex
	conns map[string]*grpc.ClientConn
	stubs map[string]InferenceClient
	addrs map[string]string
}

func NewClientPool() *ClientPool {
	return &ClientPool{
		conns: make(map[string]*grpc.ClientConn),
		stubs: make(map[string]InferenceClient),
		addrs: make(map[string]string),
	}
}

// Ensure creates or refreshes the connection for a replica.
func (p *ClientPool) Ensure(id, addr string) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	if existing, ok := p.addrs[id]; ok && existing == addr {
		return nil
	}
	if conn, ok := p.conns[id]; ok {
		_ = conn.Close()
	}

	conn, err := grpc.NewClient(
		addr,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		// Keepalive so a replica that vanishes without closing its socket --
		// a SIGKILL, a partition -- is detected in seconds rather than
		// whenever the OS eventually gives up on the TCP connection.
		grpc.WithKeepaliveParams(keepalive.ClientParameters{
			Time:                10 * time.Second,
			Timeout:             3 * time.Second,
			PermitWithoutStream: true,
		}),
	)
	if err != nil {
		return fmt.Errorf("dial %s (%s): %w", id, addr, err)
	}

	p.conns[id] = conn
	p.stubs[id] = NewInferenceClient(conn)
	p.addrs[id] = addr
	return nil
}

// Drop closes and forgets a replica's connection.
func (p *ClientPool) Drop(id string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if conn, ok := p.conns[id]; ok {
		_ = conn.Close()
	}
	delete(p.conns, id)
	delete(p.stubs, id)
	delete(p.addrs, id)
}

// Stub returns the client for a replica.
func (p *ClientPool) Stub(id string) (InferenceClient, error) {
	p.mu.RLock()
	defer p.mu.RUnlock()
	stub, ok := p.stubs[id]
	if !ok {
		return nil, fmt.Errorf("no connection for replica %s", id)
	}
	return stub, nil
}

// Health polls one replica.
func (p *ClientPool) Health(ctx context.Context, id string) (*HealthResponse, error) {
	stub, err := p.Stub(id)
	if err != nil {
		return nil, err
	}
	return stub.Health(ctx, &HealthRequest{})
}

// Close shuts every connection.
func (p *ClientPool) Close() error {
	p.mu.Lock()
	defer p.mu.Unlock()
	for id, conn := range p.conns {
		_ = conn.Close()
		delete(p.conns, id)
		delete(p.stubs, id)
		delete(p.addrs, id)
	}
	return nil
}

// SyncWith brings the pool in line with the ring's membership.
func (p *ClientPool) SyncWith(ring *Ring) {
	wanted := map[string]bool{}
	for _, rep := range ring.Snapshot() {
		wanted[rep.ID] = true
		_ = p.Ensure(rep.ID, rep.Addr)
	}

	p.mu.RLock()
	stale := make([]string, 0)
	for id := range p.conns {
		if !wanted[id] {
			stale = append(stale, id)
		}
	}
	p.mu.RUnlock()

	for _, id := range stale {
		p.Drop(id)
	}
}
