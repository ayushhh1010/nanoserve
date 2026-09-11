# Recording the failure demo

The artifact: a dashboard recording where a replica is killed at full load and
the request-outcome line never shows an error. It is the most persuasive thing
in the project because it is a claim nobody can make without having actually
built the system.

Total time: about ten minutes, most of it waiting.

---

## Before you record

**Start the stack** and let it settle:

```bash
docker compose -f deploy/docker-compose.yml up -d
```

Wait until both replicas are ready — this takes a minute or two while they load
the checkpoint:

```bash
curl -s localhost:8080/stats
```

You want `"ready": 2`. If it says `1`, wait longer.

**Open the dashboard** at <http://localhost:3000> → "Nanoserve — Router".
There is no login.

Set the time range (top right) to **Last 15 minutes** and the refresh to **5s**.
Press `Escape` if you want the menus hidden.

The two panels that tell the story are **Replicas by state** (top left) and
**Request outcomes** (top right). Make sure both are visible at once — that
pairing *is* the demo: capacity halves, errors stay at zero.

---

## The recording

**1. Start the load.** In a terminal:

```bash
python -c "import sys,time; sys.path.insert(0,'.'); from bench.chaos.harness import LoadDriver; d=LoadDriver('127.0.0.1:8080',concurrency=6,max_tokens=32); d.start(); time.sleep(300); d.stop()"
```

Leave it running. Watch the dashboard until the lines are steady — about 60
seconds. A recording that starts before the graphs have a baseline has nothing
to show a change *against*.

**2. Start your screen recorder.** On Windows, `Win+Alt+R` starts the built-in
Game Bar capture, or use ScreenToGif (<https://www.screentogif.com>) if you want
a GIF directly. Record the dashboard region only, not the whole desktop.

**3. Wait ~15 seconds** with everything steady. This is the "before" the viewer
needs.

**4. Kill a replica:**

```bash
docker kill nanoserve-replica-2
```

**5. Keep recording for ~60 seconds.** You will see, in order:

- **Replicas by state**: `ready` steps 2 → 1 within a couple of seconds
- **In-flight per replica**: one line drops to zero, the other absorbs the load
- **Retries & migrations**: a brief spike — those are the requests that were
  mid-stream on the dead replica being moved
- **Request outcomes**: `success` continues. **No error series appears.**

**6. Stop recording.** Trim to roughly 30–45 seconds. Nobody watches a
three-minute GIF.

---

## Bringing it back

```bash
docker compose -f deploy/docker-compose.yml up -d replica
```

---

## What to say about it, accurately

The honest claim is **zero dropped requests**, not a flat latency line.

Time to first token *does* rise after the kill — measured, 232 ms → 1.5 s at
p99 — because killing one of two CPU replicas removes half the fleet's
capacity, and the survivor now serves every client. That is arithmetic, not a
flaw, and claiming otherwise is the kind of thing an interviewer will probe and
find.

What the recording actually demonstrates is harder and more interesting than a
flat line: a machine disappeared mid-generation, every client that was
streaming from it kept streaming, and none of them saw an error or a duplicated
token. The p99 line recovering as capacity returns is part of the same story.

If you want the version where p99 genuinely barely moves, run three replicas
and kill one — there is then spare capacity to absorb the loss, which is how a
real fleet is provisioned. On this 4 GB card three replicas fit at
`--kv-budget-mb 192`, and `python scripts/chaos.py --only sigkill_replica`
already runs exactly that configuration: 434 requests, 0 failed, p99 TTFT
0.563 s → 0.122 s → 0.168 s across the kill.
