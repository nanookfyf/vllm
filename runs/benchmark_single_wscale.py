#!/usr/bin/env python3
"""Benchmark one elastic EP scale transition from max ep_size.

Default flow:
1. Start from ep_size=4.
2. Optionally warm up once: 4 -> target -> 4.
3. Measure 4 -> target.
4. Send one validation chat request.
5. Write one-row CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request


DEFAULT_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": "Bearer EMPTY",
}


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


def run_transition(
    *,
    ep_size: int,
    scale_url: str,
    scale_payload_base: dict,
) -> tuple[int, dict, float]:
    payload = dict(scale_payload_base)
    payload["ep_size"] = ep_size
    return post_json(scale_url, payload, DEFAULT_HEADERS)


def ensure_ok(status: int, body: dict, context: str) -> None:
    if status != 200:
        raise RuntimeError(f"{context} failed: status={status} body={body}")


def write_csv(csv_out: str, row: dict[str, object]) -> None:
    os.makedirs(os.path.dirname(csv_out) or ".", exist_ok=True)
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8005")
    parser.add_argument("--model", default="/mnt/nvme/fyf/models/DeepSeek-V2-Lite")
    parser.add_argument("--from-ep-size", type=int, default=4)
    parser.add_argument("--target-ep-size", type=int, required=True)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--dp-rank", type=int, default=0)
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--max-tokens", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--tags", nargs="+", default=["expert_weights"])
    parser.add_argument(
        "--csv-out",
        default="/mnt/nvme/fyf/proj2/vllm/runs/benchmark_single_wscale.csv",
    )
    args = parser.parse_args()

    scale_url = f"{args.base_url.rstrip('/')}/wscale"
    chat_url = f"{args.base_url.rstrip('/')}/v1/chat/completions"
    scale_payload_base = {"tags": args.tags}
    chat_headers = dict(DEFAULT_HEADERS)
    chat_headers["X-data-parallel-rank"] = str(args.dp_rank)
    chat_payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }

    # Normalize the server to the assumed initial full state.
    status, body, _ = run_transition(
        ep_size=args.from_ep_size,
        scale_url=scale_url,
        scale_payload_base=scale_payload_base,
    )
    ensure_ok(status, body, f"reset to ep_size={args.from_ep_size}")

    if args.warmup and args.from_ep_size != args.target_ep_size:
        status, body, _ = run_transition(
            ep_size=args.target_ep_size,
            scale_url=scale_url,
            scale_payload_base=scale_payload_base,
        )
        ensure_ok(status, body, f"warmup scale to ep_size={args.target_ep_size}")

        status, body, _ = post_json(chat_url, chat_payload, chat_headers)
        ensure_ok(status, body, "warmup request at target ep_size")

        status, body, _ = run_transition(
            ep_size=args.from_ep_size,
            scale_url=scale_url,
            scale_payload_base=scale_payload_base,
        )
        ensure_ok(status, body, f"warmup restore to ep_size={args.from_ep_size}")

        status, body, _ = post_json(chat_url, chat_payload, chat_headers)
        ensure_ok(status, body, "warmup request after restore")

    scale_status, scale_body, scale_time_s = run_transition(
        ep_size=args.target_ep_size,
        scale_url=scale_url,
        scale_payload_base=scale_payload_base,
    )

    request_status = 0
    request_body: dict = {}
    request_time_s = 0.0
    if scale_status == 200:
        request_status, request_body, request_time_s = post_json(
            chat_url, chat_payload, chat_headers
        )

    row = {
        "from_ep_size": args.from_ep_size,
        "target_ep_size": args.target_ep_size,
        "warmup": args.warmup,
        "action": scale_body.get("action"),
        "scale_status": scale_status,
        "scale_time_s": scale_time_s,
        "request_status": request_status,
        "request_time_s": request_time_s,
        "completion_id": request_body.get("id"),
        "finish_reason": ((request_body.get("choices") or [{}])[0]).get("finish_reason"),
        "error": None
        if scale_status == 200 and request_status == 200
        else json.dumps(
            scale_body if scale_status != 200 else request_body,
            ensure_ascii=True,
        ),
    }

    write_csv(args.csv_out, row)
    print(json.dumps(row, ensure_ascii=True))
    return 0 if row["error"] is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
