package router

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"time"

	clientv3 "go.etcd.io/etcd/client/v3"
	"go.etcd.io/etcd/client/v3/concurrency"
)

// ElectionPrefix is the etcd key space the autoscaler campaigns under.
const ElectionPrefix = "/nanoserve/election/autoscaler"

// RunAsLeader campaigns for leadership and runs fn only while holding it.
//
// Why an autoscaler needs this at all: the controller is not idempotent in the
// way a stateless request handler is. Two autoscalers observing the same
// overloaded cluster each compute the same delta and each apply it, so a
// cluster that needed one more replica gets two -- and then, seeing the
// overshoot, both scale down, and the pair oscillates. Running a single
// instance instead of electing one is not an answer: then the autoscaler is a
// single point of failure whose death is silent, because nothing scaling is
// indistinguishable from nothing needing to scale.
//
// The subtle half is giving up leadership. etcd holds the leader key under a
// session lease, so a leader that is partitioned stops renewing and loses the
// key after the TTL -- and etcd will elect someone else. If the old leader
// kept acting, both would be scaling at once, which is the exact failure the
// election was supposed to prevent, now harder to see. So fn runs under a
// context cancelled the moment the session ends, and fn must return promptly
// when it does.
func RunAsLeader(
	ctx context.Context,
	endpoints []string,
	id string,
	ttlSeconds int,
	log *slog.Logger,
	fn func(ctx context.Context),
) error {
	client, err := clientv3.New(clientv3.Config{
		Endpoints:   endpoints,
		DialTimeout: 5 * time.Second,
	})
	if err != nil {
		return fmt.Errorf("etcd connect: %w", err)
	}
	defer func() { _ = client.Close() }()

	var bo backoff
	for {
		if ctx.Err() != nil {
			return nil
		}
		if err := campaignOnce(ctx, client, id, ttlSeconds, log, fn); err != nil {
			if ctx.Err() != nil {
				return nil
			}
			// Losing a session is normal operation, not a crash: etcd
			// restarted, or this process paused long enough to miss renewals.
			// Rejoin rather than exit -- an autoscaler that exits on its first
			// hiccup leaves the cluster with no autoscaler at all.
			log.Warn("leadership attempt ended; rejoining", "err", err)
			if !sleepCtx(ctx, bo.next()) {
				return nil
			}
			continue
		}
		bo.reset()
	}
}

func campaignOnce(
	ctx context.Context,
	client *clientv3.Client,
	id string,
	ttlSeconds int,
	log *slog.Logger,
	fn func(ctx context.Context),
) error {
	session, err := concurrency.NewSession(
		client, concurrency.WithTTL(ttlSeconds), concurrency.WithContext(ctx),
	)
	if err != nil {
		return fmt.Errorf("session: %w", err)
	}
	defer func() { _ = session.Close() }()

	election := concurrency.NewElection(session, ElectionPrefix)

	log.Info("campaigning for autoscaler leadership", "id", id, "ttl_seconds", ttlSeconds)
	// Blocks until elected or ctx ends. Standby instances wait here, which is
	// what makes failover automatic: the moment the leader's lease expires,
	// etcd hands the key to the next campaigner in line.
	if err := election.Campaign(ctx, id); err != nil {
		return fmt.Errorf("campaign: %w", err)
	}
	log.Info("elected leader", "id", id)

	// Cancelled when leadership ends for any reason -- ctx, or the session
	// lease expiring underneath us.
	leaderCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	go func() {
		select {
		case <-session.Done():
			log.Warn("lost autoscaler leadership: session expired", "id", id)
			cancel()
		case <-leaderCtx.Done():
		}
	}()

	fn(leaderCtx)

	// Resign on a fresh context: leaderCtx is already cancelled on the normal
	// path, and resigning is what lets a standby take over in milliseconds
	// instead of after the full lease TTL.
	resignCtx, resignCancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer resignCancel()
	if err := election.Resign(resignCtx); err != nil && !errors.Is(err, context.Canceled) {
		log.Warn("resign failed; leadership will lapse on TTL", "err", err)
	}

	if ctx.Err() != nil {
		return nil
	}
	return errors.New("leadership ended")
}
