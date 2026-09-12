# Running Nanoserve on Kubernetes

Verified on Docker Desktop's **kind** cluster, Kubernetes v1.36.1, single node.

```
kubectl get pods -n nanoserve
NAME                          READY   STATUS    AGE
autoscaler-64cf785b9c-sltz4   1/1     Running   12s
autoscaler-64cf785b9c-wdmn9   1/1     Running   11s
etcd-0                        1/1     Running   10m
redis-8cb5485d7-ff5bj         1/1     Running   10m
replica-0                     1/1     Running   73s
replica-1                     1/1     Running   13s
router-85cdfb6d7-mrqt4        1/1     Running   12s
router-85cdfb6d7-p6sff        1/1     Running   11s
```

---

## Why there is a manual load step

`imagePullPolicy: Never`, deliberately. There is no registry in this setup, and
pointing the manifests at Docker Hub for a 1.84 GB image that changes on every
code edit would make the inner loop a push/pull cycle. The cost is that images
must be put into the node's containerd by hand.

kind's node is a *container* with its own containerd. Images in your Docker
daemon are invisible to it — `ErrImageNeverPull` is what that looks like.

## Build and load

```bash
docker compose -f deploy/docker-compose.yml build
```

Compose tags these `nanoserve/replica:dev` and `nanoserve/router:dev`, which is
exactly what the manifests reference. That is pinned in the Compose file rather
than left to Compose's default `<project>-<service>:latest` naming, because the
two drifted once and the failure was thoroughly misleading: Compose worked, the
manifests ran a three-week-old build that predated `ENV PYTHONPATH=/app`, and
the pod died with `ModuleNotFoundError: No module named 'engine'` — an error
that points at the Python path and says nothing about the real cause, which was
two names for one artifact.

Docker Desktop hides kind's node container by default. Enable
**Settings → Kubernetes → Show system containers (advanced)** first, or nothing
below can reach it.

```bash
docker save nanoserve/replica:dev | docker exec -i desktop-control-plane ctr -n k8s.io images import -
docker save nanoserve/router:dev  | docker exec -i desktop-control-plane ctr -n k8s.io images import -
```

The `-n k8s.io` namespace is not optional — containerd keeps images in
namespaces, and kubelet only looks in `k8s.io`. An import into the default
namespace succeeds, reports nothing wrong, and leaves the pod still saying
`ErrImageNeverPull`.

> **Git Bash users:** prefix with `MSYS_NO_PATHCONV=1`. MSYS rewrites arguments
> that look like Unix paths, so `/checkpoints` becomes
> `C:/Program Files/Git/checkpoints` *inside the container* — it silently
> creates a directory with that literal name and the mount still fails.

## The checkpoint

`hostPath`, because it is 321 MB of read-only data on a completely different
change cadence from the code. On kind, "the host" is the node container:

```bash
docker exec desktop-control-plane mkdir -p /checkpoints
docker cp checkpoints/run1/best.pt desktop-control-plane:/checkpoints/best.pt
```

This is the part that does not survive a `Reset cluster` — re-run it after any
cluster rebuild.

## Apply

```bash
kubectl apply -f deploy/k8s/nanoserve.yaml
kubectl get pods -n nanoserve -w
```

Replicas take 30–60s: they load the checkpoint, size the KV pool, bind the
port, and only *then* register in etcd. That ordering is the point — presence
in the registry means "can serve", not "exists".

## Verify

```bash
kubectl port-forward -n nanoserve svc/router 18080:8080
```

```bash
curl -s localhost:18080/stats
```

Both replicas should appear by their StatefulSet DNS names, which is what makes
them dialable from the router pod:

```json
{"ready":2,"replicas":[
  {"id":"replica-0","addr":"replica-0.replica.nanoserve.svc.cluster.local:9101","ready":true},
  {"id":"replica-1","addr":"replica-1.replica.nanoserve.svc.cluster.local:9101","ready":true}]}
```

```bash
curl -N localhost:18080/generate -H 'content-type: application/json' \
  -d '{"prompt":"Once upon a time there was a little cat who","max_tokens":40}'
```

Leader election, with one autoscaler active and one standing by:

```bash
kubectl logs -n nanoserve -l app=autoscaler --tail=5 | grep -i leader
```

```
msg="elected leader" id=autoscaler-64cf785b9c-sltz4-1
msg="campaigning for autoscaler leadership" id=autoscaler-64cf785b9c-wdmn9-1
```

And the etcd keyspace the whole system coordinates through:

```bash
kubectl exec -n nanoserve etcd-0 -- etcdctl get --prefix /nanoserve/ --keys-only
```

```
/nanoserve/election/autoscaler/694da0954bf9443d
/nanoserve/election/autoscaler/694da0954bf94444
/nanoserve/replicas/replica-0
/nanoserve/replicas/replica-1
/nanoserve/scale/desired
```

## Note on the kubeadm provisioner

Docker Desktop offers kubeadm and kind. The kubeadm path hung at "Starting" for
over an hour here, never binding 6443 and never creating a control-plane
container. **Reset cluster**, which also switched provisioning to kind, brought
a working cluster up in about two minutes. If kubeadm hangs, do not wait it
out.
