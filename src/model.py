"""Mathematical model of the service: a multi-server queue (Erlang C).

The service is c = WORKERS parallel servers with service time s = SERVICE_MS (fixed). Requests
arrive at rate lambda. Let a = lambda * s (offered load, in Erlangs) and rho = a / c.

Stable if rho < 1. Probability that an arriving request has to wait (Erlang C):
    C(c, a) = [a^c / (c! (1 - rho))] / [ sum_{k=0}^{c-1} a^k / k!  +  a^c / (c! (1 - rho)) ]
Mean waiting time for M/M/c:   Wq = C(c, a) * s / (c * (1 - rho))
Fixed service time (M/D/c) is approximated by Allen-Cunneen:  Wq(M/D/c) ~ Wq(M/M/c) * (1 + 0) / 2.
Mean latency   W = s + Wq        (one request, from arrival to answer)

Arrival rate seen by the workers:
    baseline : lambda = legit + attack
    proposed : lambda = legit + min(attack, rate_limit)    (the rest is answered 429 immediately)
For rho >= 1 the queue grows without bound; the real client gives up after TIMEOUT_S, so the
predicted latency is capped at the timeout.
"""
from math import factorial

WORKERS = 4
SERVICE_S = 0.010
RATE_LIMIT_RPS = 40.0
LEGIT_RPS = 20.0
TIMEOUT_S = 1.0


def erlang_c(c: int, a: float) -> float:
    rho = a / c
    top = a ** c / (factorial(c) * (1 - rho))
    return top / (sum(a ** k / factorial(k) for k in range(c)) + top)


def predicted_latency_ms(method: str, attack_rps: float, deterministic: bool = True) -> tuple[float, float]:
    """Return (rho, predicted mean latency in ms); latency capped at the client timeout."""
    lam = LEGIT_RPS + (attack_rps if method == "baseline" else min(attack_rps, RATE_LIMIT_RPS))
    a = lam * SERVICE_S
    rho = a / WORKERS
    if rho >= 1.0:
        return rho, TIMEOUT_S * 1000.0
    wq = erlang_c(WORKERS, a) * SERVICE_S / (WORKERS * (1 - rho))
    if deterministic:
        wq *= 0.5
    return rho, min(TIMEOUT_S, SERVICE_S + wq) * 1000.0
