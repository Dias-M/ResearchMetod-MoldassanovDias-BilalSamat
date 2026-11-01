"""Run the HTTP-flood experiment and write the raw log (data/raw/run_log.jsonl).

For every (method, attack level, repetition) a fresh server is started. A legitimate client sends
LEGIT_RPS requests/s (Poisson arrivals) while an attacker sends ATTACK_RPS requests/s (open loop).
Each run = WARMUP_S seconds of warm-up (logged, phase="warmup") + BENCH_S seconds of measurement.
Only legitimate requests are logged. A request that gets no answer within TIMEOUT_S is logged with
status 599 and latency = TIMEOUT_S (it is NOT dropped). Run order is shuffled (seed) to spread drift.

  python src/run_experiment.py --reps 5          # full run, about 12 minutes
  python src/run_experiment.py --reps 1 --bench 3 --out data/raw/smoke.jsonl   # quick check
"""
import argparse
import asyncio
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PORT = 18080
LEVELS = [0, 100, 200, 300, 400, 600]     # attack requests/s
LEGIT_RPS = 20.0
TIMEOUT_S = 1.0


async def one_request(client_id: str, timeout: float) -> tuple[float, int]:
    t0 = time.perf_counter()
    status = 599
    try:
        async def call() -> int:
            r, w = await asyncio.open_connection("127.0.0.1", PORT)
            w.write(f"GET / HTTP/1.1\r\nHost: x\r\nX-Client-Id: {client_id}\r\n\r\n".encode())
            await w.drain()
            first = await r.readline()
            w.close()
            return int(first.split()[1])
        status = await asyncio.wait_for(call(), timeout)
    except (asyncio.TimeoutError, OSError, ValueError, IndexError):
        status = 599
    return (time.perf_counter() - t0) * 1000.0, status


async def drive(attack_rps: int, warmup_s: float, bench_s: float, rng: random.Random) -> list[dict]:
    records: list[dict] = []
    total = warmup_s + bench_s
    t_start = time.perf_counter()
    pending: set[asyncio.Task] = set()

    async def attacker() -> None:
        gap = 1.0 / attack_rps
        nxt = time.perf_counter()
        while nxt - t_start < total:
            nxt += gap
            t = asyncio.create_task(one_request("attacker", TIMEOUT_S))
            pending.add(t)
            t.add_done_callback(pending.discard)
            await asyncio.sleep(max(0.0, nxt - time.perf_counter()))

    async def legit() -> None:
        seq, nxt = 0, time.perf_counter()
        while nxt - t_start < total:
            nxt += rng.expovariate(LEGIT_RPS)
            await asyncio.sleep(max(0.0, nxt - time.perf_counter()))
            phase = "warmup" if nxt - t_start < warmup_s else "benchmark"

            async def go(s=seq, ph=phase) -> None:
                lat, st = await one_request("legit", TIMEOUT_S)
                records.append(dict(phase=ph, seq=s, latency_ms=round(lat, 3), status=st))
            t = asyncio.create_task(go())
            pending.add(t)
            t.add_done_callback(pending.discard)
            seq += 1

    jobs = [legit()] + ([attacker()] if attack_rps > 0 else [])
    await asyncio.gather(*jobs)
    if pending:
        await asyncio.wait(pending, timeout=TIMEOUT_S + 1)
    records.sort(key=lambda r: r["seq"])
    # sequence numbers restart per phase so that "first N of a run" is meaningful
    for ph in ("warmup", "benchmark"):
        for i, r in enumerate([r for r in records if r["phase"] == ph]):
            r["seq"] = i
    return records


def run_one(method: str, attack_rps: int, warmup_s: float, bench_s: float, seed: int) -> list[dict]:
    srv = subprocess.Popen([sys.executable, str(HERE / "server.py"), "--mode", method, "--port", str(PORT)])
    try:
        time.sleep(0.8)
        return asyncio.run(drive(attack_rps, warmup_s, bench_s, random.Random(seed)))
    finally:
        srv.terminate()
        srv.wait()
        time.sleep(1.0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warmup", type=float, default=2.0)
    ap.add_argument("--bench", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/raw/run_log.jsonl")
    a = ap.parse_args()
    plan = [(m, lv, rep) for rep in range(a.reps) for m in ("baseline", "proposed") for lv in LEVELS]
    random.Random(a.seed).shuffle(plan)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    env = dict(python=platform.python_version(), platform=platform.platform(), cpus=__import__("os").cpu_count(),
               legit_rps=LEGIT_RPS, timeout_s=TIMEOUT_S, warmup_s=a.warmup, bench_s=a.bench, seed=a.seed)
    (out.parent / "environment.json").write_text(json.dumps(env, indent=2))
    with out.open("w") as f:
        for i, (method, lv, rep) in enumerate(plan, 1):
            t0 = time.time()
            recs = run_one(method, lv, a.warmup, a.bench, a.seed * 1000 + i)
            for r in recs:
                f.write(json.dumps(dict(run=i, rep=rep, method=method, attack_rps=lv, **r)) + "\n")
            f.flush()
            ok = sum(r["status"] == 200 for r in recs if r["phase"] == "benchmark")
            print(f"[{i}/{len(plan)}] {method:8s} attack={lv:4d} rep={rep} legit_ok={ok}/"
                  f"{sum(r['phase'] == 'benchmark' for r in recs)} {time.time() - t0:.1f}s", flush=True)
