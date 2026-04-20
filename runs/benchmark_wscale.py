#!/usr/bin/env python3
"""Benchmark elastic EP scale transitions via the /wscale API.

Example:
  python3 runs/benchmark_wscale.py \
    --base-url http://127.0.0.1:8005 \
    --model /mnt/nvme/fyf/models/DeepSeek-V2-Lite \
    --mode custom \
    --sequence 4,3,2,1,2,3,4 \
    --csv-out /tmp/wscale.csv \
    --repeats 3
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, asdict


DEFAULT_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": "Bearer EMPTY",
}


@dataclass
class StepResult:
    iteration: int
    step: int
    from_ep_size: int | None
    ep_size: int
    action: str
    scale_status: int
    scale_time_s: float
    request_status: int
    request_time_s: float
    completion_id: str | None
    finish_reason: str | None
    error: str | None


def post_json(url: str, payload: dict, headers: dict[str, str]) -> tuple[int, dict, float]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            elapsed = time.perf_counter() - start
            body = response.read().decode("utf-8")
            return response.status, json.loads(body) if body else {}, elapsed
    except urllib.error.HTTPError as exc:
        elapsed = time.perf_counter() - start
        body = exc.read().decode("utf-8")
        parsed = {"raw": body}
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            pass
        return exc.code, parsed, elapsed


def run_benchmark(args: argparse.Namespace) -> list[StepResult]:
    sequence = build_sequence(args)
    if not sequence:
        raise ValueError("sequence must not be empty")

    scale_url = f"{args.base_url.rstrip('/')}/wscale"
    chat_url = f"{args.base_url.rstrip('/')}/v1/chat/completions"
    chat_headers = dict(DEFAULT_HEADERS)
    chat_headers["X-data-parallel-rank"] = str(args.dp_rank)

    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    scale_payload_base = {"tags": args.tags}

    results: list[StepResult] = []
    for iteration in range(1, args.repeats + 1):
        prev_ep_size = args.initial_ep_size
        for step, ep_size in enumerate(sequence, start=1):
            if args.warmup_each_case and prev_ep_size != ep_size:
                warmup_transition(
                    from_ep_size=prev_ep_size,
                    to_ep_size=ep_size,
                    scale_url=scale_url,
                    chat_url=chat_url,
                    scale_payload_base=scale_payload_base,
                    payload=payload,
                    chat_headers=chat_headers,
                )

            scale_payload = dict(scale_payload_base)
            scale_payload["ep_size"] = ep_size
            scale_status, scale_body, scale_time_s = post_json(
                scale_url, scale_payload, DEFAULT_HEADERS
            )
            action = str(scale_body.get("action", "unknown"))

            completion_id = None
            finish_reason = None
            request_error = None
            request_status = 0
            request_time_s = 0.0
            if scale_status == 200:
                request_status, request_body, request_time_s = post_json(
                    chat_url, payload, chat_headers
                )
                if request_status == 200:
                    completion_id = request_body.get("id")
                    choices = request_body.get("choices") or []
                    if choices:
                        finish_reason = choices[0].get("finish_reason")
                else:
                    request_error = json.dumps(request_body, ensure_ascii=True)
            else:
                request_error = json.dumps(scale_body, ensure_ascii=True)

            result = StepResult(
                iteration=iteration,
                step=step,
                from_ep_size=prev_ep_size,
                ep_size=ep_size,
                action=action,
                scale_status=scale_status,
                scale_time_s=scale_time_s,
                request_status=request_status,
                request_time_s=request_time_s,
                completion_id=completion_id,
                finish_reason=finish_reason,
                error=request_error,
            )
            results.append(result)
            print(
                json.dumps(asdict(result), ensure_ascii=True),
                flush=True,
            )
            prev_ep_size = ep_size
            if result.error and args.stop_on_error:
                return results
    return results


def warmup_transition(
    *,
    from_ep_size: int,
    to_ep_size: int,
    scale_url: str,
    chat_url: str,
    scale_payload_base: dict,
    payload: dict,
    chat_headers: dict[str, str],
) -> None:
    print(
        json.dumps(
            {
                "phase": "warmup",
                "from_ep_size": from_ep_size,
                "to_ep_size": to_ep_size,
            },
            ensure_ascii=True,
        ),
        flush=True,
    )
    run_transition(
        target_ep_size=to_ep_size,
        scale_url=scale_url,
        chat_url=chat_url,
        scale_payload_base=scale_payload_base,
        payload=payload,
        chat_headers=chat_headers,
    )
    run_transition(
        target_ep_size=from_ep_size,
        scale_url=scale_url,
        chat_url=chat_url,
        scale_payload_base=scale_payload_base,
        payload=payload,
        chat_headers=chat_headers,
    )


def run_transition(
    *,
    target_ep_size: int,
    scale_url: str,
    chat_url: str,
    scale_payload_base: dict,
    payload: dict,
    chat_headers: dict[str, str],
) -> None:
    scale_payload = dict(scale_payload_base)
    scale_payload["ep_size"] = target_ep_size
    scale_status, scale_body, _ = post_json(scale_url, scale_payload, DEFAULT_HEADERS)
    if scale_status != 200:
        raise RuntimeError(
            f"warmup wscale failed for ep_size={target_ep_size}: {scale_body}"
        )
    request_status, request_body, _ = post_json(chat_url, payload, chat_headers)
    if request_status != 200:
        raise RuntimeError(
            f"warmup request failed for ep_size={target_ep_size}: {request_body}"
        )


def build_sequence(args: argparse.Namespace) -> list[int]:
    if args.mode == "custom":
        return [int(part) for part in args.sequence.split(",") if part.strip()]
    if args.max_ep_size <= 0:
        raise ValueError("max_ep_size must be positive")
    if args.mode == "walk":
        return list(range(1, args.max_ep_size + 1))
    if args.mode == "roundtrip":
        up = list(range(1, args.max_ep_size + 1))
        down = list(range(args.max_ep_size - 1, 0, -1))
        return up + down
    if args.mode == "all-pairs":
        sequence: list[int] = []
        for src in range(1, args.max_ep_size + 1):
            sequence.append(src)
            for dst in range(1, args.max_ep_size + 1):
                if dst != src:
                    sequence.append(dst)
        return sequence
    raise ValueError(f"Unsupported mode: {args.mode}")


def write_csv(results: list[StepResult], csv_out: str) -> None:
    if not csv_out:
        return
    os.makedirs(os.path.dirname(csv_out) or ".", exist_ok=True)
    fieldnames = list(asdict(results[0]).keys()) if results else list(
        StepResult(
            iteration=0,
            step=0,
            from_ep_size=None,
            ep_size=0,
            action="",
            scale_status=0,
            scale_time_s=0.0,
            request_status=0,
            request_time_s=0.0,
            completion_id=None,
            finish_reason=None,
            error=None,
        ).__dict__.keys()
    )
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def print_summary(results: list[StepResult]) -> None:
    successful = [r for r in results if r.scale_status == 200 and r.request_status == 200]
    if not successful:
        print("\nNo successful benchmark steps.", file=sys.stderr)
        return

    print("\nSummary")
    print("from->to  count  scale_avg_s  scale_p95_s  req_avg_s  req_p95_s")
    by_transition: dict[tuple[int | None, int], list[StepResult]] = {}
    for result in successful:
        key = (result.from_ep_size, result.ep_size)
        by_transition.setdefault(key, []).append(result)

    for (from_ep_size, ep_size) in sorted(by_transition):
        group = by_transition[(from_ep_size, ep_size)]
        scale_times = [r.scale_time_s for r in group]
        req_times = [r.request_time_s for r in group]
        print(
            f"{str(from_ep_size) + '->' + str(ep_size):<8s} {len(group):<5d} "
            f"{statistics.mean(scale_times):<12.4f} "
            f"{percentile(scale_times, 95):<12.4f} "
            f"{statistics.mean(req_times):<10.4f} "
            f"{percentile(req_times, 95):<10.4f}"
        )


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (pct / 100.0)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8005",
        help="vLLM server base URL",
    )
    parser.add_argument(
        "--model",
        default="/mnt/nvme/fyf/models/DeepSeek-V2-Lite",
        help="Model name/path passed to chat completions",
    )
    parser.add_argument(
        "--mode",
        choices=("custom", "walk", "roundtrip", "all-pairs"),
        default="all-pairs",
        help="How to generate ep_size targets",
    )
    parser.add_argument(
        "--sequence",
        default="4,3,2,1,2,3,4",
        help="Comma-separated ep_size targets used when --mode=custom",
    )
    parser.add_argument(
        "--max-ep-size",
        type=int,
        default=4,
        help="Maximum ep_size used by generated modes",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of times to replay the sequence",
    )
    parser.add_argument(
        "--initial-ep-size",
        type=int,
        default=4,
        help="Known starting ep_size before the first measured transition",
    )
    parser.add_argument(
        "--dp-rank",
        type=int,
        default=0,
        help="X-data-parallel-rank used for the validation request",
    )
    parser.add_argument(
        "--prompt",
        default="Hello",
        help="Prompt sent after each scale transition",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=5,
        help="max_tokens for the validation request",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="temperature for the validation request",
    )
    parser.add_argument(
        "--tags",
        nargs="+",
        default=["expert_weights"],
        help="Tags passed to /wscale",
    )
    parser.add_argument(
        "--csv-out",
        default="/mnt/nvme/fyf/proj2/vllm/runs/benchmark_wscale.csv",
        help="Write raw per-step results to this CSV path",
    )
    parser.add_argument(
        "--warmup-each-case",
        action="store_true",
        help="Before each measured transition, run target once and restore, then record the second run",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately if a scale or validation request fails",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = run_benchmark(args)
    write_csv(results, args.csv_out)
    print_summary(results)
    failed = any(r.scale_status != 200 or r.request_status != 200 for r in results)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
