"""Target web service for the HTTP-flood experiment (asyncio, standard library only).

The service has WORKERS parallel workers; handling one request takes SERVICE_MS milliseconds
(a stand-in for application work). Requests wait in a FIFO queue for a free worker.

  --mode baseline : no protection, every request is queued for a worker.
  --mode proposed : per-client token bucket (rate limiting). A request over the limit is answered
                    at once with 429 and never reaches a worker. The client is identified by the
                    X-Client-Id header (on a real deployment: source IP / API key).
"""
import argparse
import asyncio
import time

WORKERS = 4
SERVICE_MS = 10.0
RATE_LIMIT_RPS = 40.0   # tokens per second per client (proposed mode)
BURST = 20.0            # bucket size


class Bucket:
    def __init__(self) -> None:
        self.tokens, self.t = BURST, time.monotonic()

    def allow(self) -> bool:
        now = time.monotonic()
        self.tokens = min(BURST, self.tokens + (now - self.t) * RATE_LIMIT_RPS)
        self.t = now
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False


async def main(mode: str, port: int) -> None:
    workers = asyncio.Semaphore(WORKERS)
    buckets: dict[str, Bucket] = {}

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        client = "?"
        try:
            await reader.readline()                       # request line
            while True:                                   # headers
                line = await reader.readline()
                if line in (b"\r\n", b"\n", b""):
                    break
                if line.lower().startswith(b"x-client-id:"):
                    client = line.split(b":", 1)[1].strip().decode()
            if mode == "proposed" and not buckets.setdefault(client, Bucket()).allow():
                status = b"429 Too Many Requests"
            else:
                async with workers:
                    await asyncio.sleep(SERVICE_MS / 1000)
                status = b"200 OK"
            writer.write(b"HTTP/1.1 " + status + b"\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok")
            await writer.drain()
        except (ConnectionError, asyncio.CancelledError):
            pass
        finally:
            writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", port, backlog=4096)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "proposed"], required=True)
    ap.add_argument("--port", type=int, default=8080)
    a = ap.parse_args()
    asyncio.run(main(a.mode, a.port))
