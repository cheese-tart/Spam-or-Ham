"""Measure client-side latency and throughput for the running prediction API."""

import argparse
import json
import math
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPException
from time import perf_counter
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PAYLOADS = tuple(
    json.dumps(message).encode("utf-8")
    for message in (
        {"text": "Hi, are we still meeting tomorrow?\nThanks!", "source": "email"},
        {"text": "Congratulations! Claim your FREE cash prize now!", "source": "sms"},
    )
)


def measure_prediction(url: str, payload: bytes, timeout: float) -> float | str:
    try:
        request = Request(url, data=payload, headers={"Content-Type": "application/json"})
        started = perf_counter()
        with urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                return f"HTTP {response.status}"
            body = response.read()
        elapsed_ms = (perf_counter() - started) * 1_000
    except HTTPError as error:
        error.close()
        return f"HTTP {error.code}"
    except URLError as error:
        return str(error.reason)
    except (HTTPException, OSError, ValueError) as error:
        return str(error)

    try:
        prediction = json.loads(body)
    except ValueError:
        return "Response body is not valid JSON"
    if not isinstance(prediction, dict) or prediction.get("label") not in ("ham", "spam"):
        return "Response is missing a valid prediction label"
    return elapsed_ms


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000/predict")
    parser.add_argument("--requests", type=int, default=200, help="Number of measured requests")
    parser.add_argument("--warmup", type=int, default=10, help="Number of unmeasured requests")
    parser.add_argument("--concurrency", type=int, default=1, help="Maximum simultaneous requests")
    parser.add_argument("--timeout", type=float, default=5.0, help="HTTP timeout in seconds")
    args = parser.parse_args()
    if args.requests < 1 or args.concurrency < 1 or args.warmup < 0:
        parser.error("requests and concurrency must be positive, and warmup must be nonnegative")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be a positive, finite number")

    def run_request(index: int) -> float | str:
        payload = PAYLOADS[index % len(PAYLOADS)]
        return measure_prediction(args.url, payload, args.timeout)

    latencies_ms = []
    errors = Counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for outcome in pool.map(run_request, range(args.warmup)):
            if isinstance(outcome, str):
                print(f"Warmup failed for {args.url} ({outcome})")
                return 1

        started = perf_counter()
        for outcome in pool.map(run_request, range(args.requests)):
            if isinstance(outcome, str):
                errors[outcome] += 1
            else:
                latencies_ms.append(outcome)
        elapsed = perf_counter() - started

    successes = len(latencies_ms)
    failures = args.requests - successes
    print(f"Endpoint      {args.url}")
    print(f"Concurrency   {args.concurrency}")
    print(f"Warmup        {args.warmup} requests")
    print(f"Measured      {args.requests} requests in {elapsed:.2f} s")
    print(f"Throughput    {successes / elapsed:.2f} successful requests/s")
    print(f"Errors        {failures}/{args.requests} ({failures / args.requests:.1%})")

    if latencies_ms:
        latencies_ms.sort()
        for percentile in (50, 95):
            index = math.ceil(percentile * successes / 100) - 1
            print(f"p{percentile} latency   {latencies_ms[index]:.2f} ms")

    for error, count in errors.most_common():
        print(f"  {count} request(s) failed ({error})")
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
