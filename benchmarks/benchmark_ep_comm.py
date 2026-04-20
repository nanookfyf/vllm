#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark EP communication backends with torchrun.

This script compares two EP communication styles used by vLLM:

1. `agrs`: NCCL collective path mirroring `AgRsAll2AllManager`
   (`all_gather` for dispatch, `reduce_scatter` for combine).
2. `nixl_ep`: NIXL low-latency point-to-point EP path
   (`nixl_ep.Buffer.dispatch/combine`).

The two paths are not perfectly identical semantically:
- `agrs` is a collective gather/scatter microbenchmark on the raw payloads.
- `nixl_ep` exercises the real token-routing EP dispatch/combine kernels.

Use the latency numbers as the primary comparison signal. Payload-based
bandwidth estimates are provided to help normalize trends.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import itertools
import json
import os
import statistics
import sys
from dataclasses import asdict, dataclass
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


NIXL_SUPPORTED_HIDDEN_SIZES = {2048, 2560, 3072, 4096, 5120, 6144, 7168, 8192}
# NIXL EP low-latency dispatch kernels enforce a num_topk upper bound.
# The current installed package errors out at topk=16; local source shows 11.
NIXL_MAX_TOPK = 11
# Older installed nixl_ep packages can underestimate RDMA sizing for
# variable-size routing cases. Try a small set of escalating safety factors
# and keep the first one that is sufficient, to avoid over-allocating and
# hitting OOM on larger hidden sizes.
NIXL_RDMA_SIZE_SAFETY_FACTORS = (1, 2, 4)
DEFAULT_SWEEP_NUM_TOKENS = [32, 64, 128, 256, 512]
DEFAULT_SWEEP_HIDDEN_SIZES = [2048, 4096, 5120, 7168, 8192]
DEFAULT_SWEEP_NUM_EXPERTS = [8, 16, 32, 64]
DEFAULT_SWEEP_TOPKS = [1, 2, 4, 8]


@dataclass
class BenchStats:
    backend: str
    mode: str
    size_mode: str
    world_size: int
    rank: int
    num_tokens: int
    hidden_size: int
    num_experts: int
    topk: int
    dtype: str
    warmup_iters: int
    measure_iters: int
    avg_us: float
    p50_us: float
    p95_us: float
    min_us: float
    max_us: float
    estimated_payload_bytes_per_rank: int | None
    estimated_bandwidth_gbps: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=["agrs", "agrs_torch", "nixl_ep", "both", "all"],
        default="both",
        help="Communication backend to benchmark.",
    )
    parser.add_argument(
        "--mode",
        choices=["dispatch", "combine", "end2end"],
        default="end2end",
        help="Benchmark stage to run.",
    )
    parser.add_argument("--num-tokens", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=7168)
    parser.add_argument("--num-experts", type=int, default=32)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument(
        "--max-tokens-per-rank",
        type=int,
        default=None,
        help="Max dispatch tokens per rank for NIXL. Defaults to --num-tokens.",
    )
    parser.add_argument(
        "--dtype",
        choices=["bf16", "fp16"],
        default="bf16",
    )
    parser.add_argument("--warmup-iters", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--routing",
        choices=["uniform", "skewed"],
        default="uniform",
        help="Synthetic top-k routing pattern for NIXL.",
    )
    parser.add_argument(
        "--nixl-tcp-store-port",
        type=int,
        default=None,
        help="TCPStore port for NIXL metadata. Defaults to MASTER_PORT + 1.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON summary only.",
    )
    parser.add_argument(
        "--size-mode",
        choices=["equal", "variable"],
        default="equal",
        help="Whether each rank uses equal token counts or variable token counts.",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Sweep common parameter grids and optionally write CSV.",
    )
    parser.add_argument(
        "--csv-out",
        type=str,
        default=None,
        help="CSV output path for sweep results.",
    )
    parser.add_argument(
        "--sweep-num-tokens",
        type=str,
        default=None,
        help="Comma-separated token counts. Default: common set.",
    )
    parser.add_argument(
        "--sweep-hidden-sizes",
        type=str,
        default=None,
        help="Comma-separated hidden sizes. Default: common set.",
    )
    parser.add_argument(
        "--sweep-num-experts",
        type=str,
        default=None,
        help="Comma-separated expert counts. Default: common set.",
    )
    parser.add_argument(
        "--sweep-topks",
        type=str,
        default=None,
        help="Comma-separated topk values. Default: common set.",
    )
    return parser.parse_args()


def get_rank_env() -> tuple[int, int, int]:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    return rank, world_size, local_rank


def dtype_from_name(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16}[name]


def init_distributed(local_rank: int) -> tuple[int, int]:
    torch.cuda.set_device(local_rank)
    kwargs: dict[str, Any] = {"backend": "nccl"}
    if "device_id" in inspect.signature(dist.init_process_group).parameters:
        kwargs["device_id"] = torch.device("cuda", local_rank)
    dist.init_process_group(**kwargs)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size


def barrier() -> None:
    barrier_sig = inspect.signature(dist.barrier)
    if "device_ids" in barrier_sig.parameters:
        dist.barrier(device_ids=[torch.cuda.current_device()])
    else:
        dist.barrier()
    torch.cuda.synchronize()


def build_owner_table(num_experts: int, world_size: int, device: torch.device) -> torch.Tensor:
    assert num_experts % world_size == 0, (
        f"num_experts ({num_experts}) must be divisible by world_size ({world_size})"
    )
    experts_per_rank = num_experts // world_size
    return torch.arange(num_experts, device=device, dtype=torch.int64) // experts_per_rank


def make_topk_ids(
    num_tokens: int,
    topk: int,
    num_experts: int,
    world_size: int,
    rank: int,
    routing: str,
    device: torch.device,
) -> torch.Tensor:
    owner = build_owner_table(num_experts, world_size, device)
    if routing == "uniform":
        return torch.randint(
            0, num_experts, (num_tokens, topk), device=device, dtype=torch.int64
        )

    experts_per_rank = num_experts // world_size
    local_start = rank * experts_per_rank
    hot_remote_rank = (rank + 1) % world_size
    hot_remote_start = hot_remote_rank * experts_per_rank

    ids = torch.empty((num_tokens, topk), device=device, dtype=torch.int64)
    ids[:, 0] = torch.randint(
        local_start,
        local_start + experts_per_rank,
        (num_tokens,),
        device=device,
        dtype=torch.int64,
    )
    for k in range(1, topk):
        ids[:, k] = torch.randint(
            hot_remote_start,
            hot_remote_start + experts_per_rank,
            (num_tokens,),
            device=device,
            dtype=torch.int64,
        )
    # Force deduped rank destinations per token where possible.
    owners = owner[ids]
    for k in range(1, topk):
        dup = owners[:, k] == owners[:, 0]
        if dup.any():
            ids[dup, k] = (ids[dup, k] + experts_per_rank) % num_experts
    return ids


def estimate_nixl_remote_payload_bytes(
    topk_ids: torch.Tensor,
    hidden_size: int,
    dtype: torch.dtype,
    rank: int,
    world_size: int,
    num_experts: int,
) -> int:
    owner = build_owner_table(num_experts, world_size, topk_ids.device)
    owners = owner[topk_ids]
    remote = owners != rank
    unique_remote = torch.zeros(
        (topk_ids.size(0), world_size), device=topk_ids.device, dtype=torch.bool
    )
    unique_remote.scatter_(1, owners.clamp_min(0), remote)
    remote_copies = int(unique_remote.sum().item())
    return remote_copies * hidden_size * torch.tensor([], dtype=dtype).element_size()


def percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    idx = (len(sorted_values) - 1) * pct
    lo = int(idx)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = idx - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def parse_int_list(value: str | None, default: list[int]) -> list[int]:
    if value is None:
        return list(default)
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def make_case_args(
    args: argparse.Namespace,
    *,
    num_tokens: int,
    hidden_size: int,
    num_experts: int,
    topk: int,
) -> argparse.Namespace:
    data = vars(args).copy()
    data["num_tokens"] = num_tokens
    data["hidden_size"] = hidden_size
    data["num_experts"] = num_experts
    data["topk"] = topk
    return argparse.Namespace(**data)


def build_rank_sizes(num_tokens: int, world_size: int, size_mode: str) -> list[int]:
    if size_mode == "equal":
        return [num_tokens] * world_size
    step = max(1, num_tokens // 16)
    center = (world_size - 1) / 2.0
    sizes = []
    for rank in range(world_size):
        size = int(round(num_tokens + (rank - center) * step))
        sizes.append(max(1, size))
    return sizes


def iter_sweep_cases(args: argparse.Namespace) -> list[argparse.Namespace]:
    token_grid = parse_int_list(args.sweep_num_tokens, DEFAULT_SWEEP_NUM_TOKENS)
    hidden_grid = parse_int_list(
        args.sweep_hidden_sizes, DEFAULT_SWEEP_HIDDEN_SIZES
    )
    expert_grid = parse_int_list(
        args.sweep_num_experts, DEFAULT_SWEEP_NUM_EXPERTS
    )
    topk_grid = parse_int_list(args.sweep_topks, DEFAULT_SWEEP_TOPKS)
    cases = []
    for num_tokens, hidden_size, num_experts, topk in itertools.product(
        token_grid, hidden_grid, expert_grid, topk_grid
    ):
        cases.append(
            make_case_args(
                args,
                num_tokens=num_tokens,
                hidden_size=hidden_size,
                num_experts=num_experts,
                topk=topk,
            )
        )
    return cases


def is_valid_case(
    args: argparse.Namespace,
    world_size: int,
) -> tuple[bool, str | None]:
    if args.num_experts < world_size:
        return False, "num_experts < world_size"
    if args.num_experts % world_size != 0:
        return False, "num_experts not divisible by world_size"
    if args.topk > args.num_experts:
        return False, "topk > num_experts"
    if args.max_tokens_per_rank is not None and args.max_tokens_per_rank < args.num_tokens:
        return False, "max_tokens_per_rank < num_tokens"
    return True, None


def is_valid_case_for_backend(
    args: argparse.Namespace,
    world_size: int,
    backend: str,
) -> tuple[bool, str | None]:
    ok, reason = is_valid_case(args, world_size)
    if not ok:
        return ok, reason
    if backend == "nixl_ep":
        if args.hidden_size not in NIXL_SUPPORTED_HIDDEN_SIZES:
            return False, f"hidden_size unsupported by nixl_ep"
        if args.topk > NIXL_MAX_TOPK:
            return False, f"topk>{NIXL_MAX_TOPK} unsupported by nixl_ep"
    return True, None


def collect_stats(
    backend: str,
    mode: str,
    rank: int,
    world_size: int,
    args: argparse.Namespace,
    lat_us: list[float],
    payload_bytes: int | None,
) -> BenchStats:
    lat_sorted = sorted(lat_us)
    avg_us = statistics.fmean(lat_us)
    bw_gbps = None
    if payload_bytes is not None and avg_us > 0:
        bw_gbps = payload_bytes / avg_us / 1e3
    return BenchStats(
        backend=backend,
        mode=mode,
        size_mode=args.size_mode,
        world_size=world_size,
        rank=rank,
        num_tokens=args.num_tokens,
        hidden_size=args.hidden_size,
        num_experts=args.num_experts,
        topk=args.topk,
        dtype=args.dtype,
        warmup_iters=args.warmup_iters,
        measure_iters=args.iters,
        avg_us=avg_us,
        p50_us=percentile(lat_sorted, 0.50),
        p95_us=percentile(lat_sorted, 0.95),
        min_us=min(lat_us),
        max_us=max(lat_us),
        estimated_payload_bytes_per_rank=payload_bytes,
        estimated_bandwidth_gbps=bw_gbps,
    )


def benchmark_agrs(
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> BenchStats:
    return benchmark_agrs_torch(args, rank, world_size, device, dtype)


def benchmark_agrs_torch(
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> BenchStats:
    sizes = build_rank_sizes(args.num_tokens, world_size, args.size_mode)
    local_tokens = sizes[rank]
    hidden = torch.randn(
        local_tokens, args.hidden_size, device=device, dtype=dtype
    )
    gathered = torch.empty(
        (sum(sizes), args.hidden_size), device=device, dtype=dtype
    )
    combined_input = torch.randn_like(gathered_hidden)
    combined_out = torch.empty_like(hidden)

    payload_bytes = (
        (sum(sizes) - local_tokens) * args.hidden_size * hidden.element_size()
    )

    def dispatch() -> None:
        if args.size_mode == "equal":
            dist.all_gather_into_tensor(gathered, hidden)
        else:
            chunks = list(gathered.split(sizes, dim=0))
            dist.all_gather(chunks, hidden)

    def combine() -> None:
        if args.size_mode == "equal":
            dist.reduce_scatter_tensor(combined_out, combined_input)
        else:
            input_chunks = list(combined_input.split(sizes, dim=0))
            dist.reduce_scatter(combined_out, input_chunks)

    if args.mode == "dispatch":
        fn = dispatch
    elif args.mode == "combine":
        fn = combine
    else:
        def fn() -> None:
            dispatch()
            combine()

        payload_bytes *= 2

    return run_timed_loop(
        "agrs_torch", args.mode, fn, args, rank, world_size, payload_bytes
    )


def benchmark_agrs_pynccl(
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> BenchStats:
    from vllm.distributed.device_communicators.cuda_communicator import (
        CudaCommunicator,
    )

    gloo_group = dist.new_group(backend="gloo")
    cuda_comm = CudaCommunicator(
        cpu_group=gloo_group,
        device=device,
        unique_name="bench_ep:0",
    )
    pynccl = cuda_comm.pynccl_comm
    assert pynccl is not None and pynccl.available and not pynccl.disabled
    sizes = build_rank_sizes(args.num_tokens, world_size, args.size_mode)
    local_tokens = sizes[rank]

    hidden = torch.randn(
        local_tokens, args.hidden_size, device=device, dtype=dtype
    )
    topk_weights = torch.rand(
        local_tokens, args.topk, device=device, dtype=torch.float32
    )
    topk_ids = make_topk_ids(
        local_tokens,
        args.topk,
        args.num_experts,
        world_size,
        rank,
        args.routing,
        device,
    ).to(torch.int64)
    gathered_hidden = torch.empty(
        (sum(sizes), args.hidden_size), device=device, dtype=dtype
    )
    gathered_topk_weights = torch.empty(
        (sum(sizes), args.topk), device=device, dtype=torch.float32
    )
    gathered_topk_ids = torch.empty(
        (sum(sizes), args.topk), device=device, dtype=torch.int64
    )
    combined_input = torch.randn_like(gathered_hidden)
    combined_out = torch.empty_like(hidden)

    hidden_payload = (sum(sizes) - local_tokens) * args.hidden_size * hidden.element_size()
    topk_weight_payload = (
        (sum(sizes) - local_tokens) * args.topk * topk_weights.element_size()
    )
    topk_id_payload = (
        (sum(sizes) - local_tokens) * args.topk * topk_ids.element_size()
    )
    payload_bytes = hidden_payload + topk_weight_payload + topk_id_payload

    def dispatch() -> None:
        gathered = cuda_comm.all_gatherv(
            [hidden, topk_weights, topk_ids], dim=0, sizes=sizes
        )
        gathered_hidden.copy_(gathered[0])
        gathered_topk_weights.copy_(gathered[1])
        gathered_topk_ids.copy_(gathered[2])

    def combine() -> None:
        combined_out.copy_(cuda_comm.reduce_scatterv(combined_input, dim=0, sizes=sizes))

    if args.mode == "dispatch":
        fn = dispatch
    elif args.mode == "combine":
        payload_bytes = hidden_payload
        fn = combine
    else:

        def fn() -> None:
            dispatch()
            combine()

        payload_bytes += hidden_payload

    try:
        return run_timed_loop(
            "agrs", args.mode, fn, args, rank, world_size, payload_bytes
        )
    finally:
        barrier()
        cuda_comm.destroy()
        dist.destroy_process_group(gloo_group)


def create_nixl_buffer(
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    max_tokens: int,
    rdma_safety_factor: int = 1,
):
    import nixl_ep

    num_experts = args.num_experts
    experts_per_rank = num_experts // world_size
    max_num_ep_ranks = max(world_size, int(os.environ.get("VLLM_NIXL_EP_MAX_NUM_RANKS", "32")))
    max_num_global_experts = max_num_ep_ranks * experts_per_rank
    master_addr = os.environ["MASTER_ADDR"]
    master_port = int(os.environ["MASTER_PORT"])
    tcp_port = args.nixl_tcp_store_port or (master_port + 1)

    tcp_store = dist.TCPStore(
        host_name=master_addr,
        port=tcp_port,
        world_size=world_size,
        is_master=(rank == 0),
        timeout=timedelta(seconds=300),
        wait_for_workers=True,
    )
    buffer = nixl_ep.Buffer(
        rank=rank,
        explicitly_destroy=True,
        tcp_store_group=tcp_store,
    )
    rdma_bytes = nixl_ep.Buffer.get_rdma_size_hint(
        max_tokens, args.hidden_size, max_num_ep_ranks, max_num_global_experts
    )
    rdma_bytes *= rdma_safety_factor
    buffer.update_memory_buffers(
        num_ranks=max_num_ep_ranks,
        num_experts_per_rank=experts_per_rank,
        num_rdma_bytes=rdma_bytes,
    )
    buffer.connect_ranks(list(range(world_size)))
    return buffer


def benchmark_nixl_ep(
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> BenchStats:
    import nixl_ep

    sizes = build_rank_sizes(args.num_tokens, world_size, args.size_mode)
    local_tokens = sizes[rank]
    max_tokens = args.max_tokens_per_rank or max(sizes)
    x = torch.randn(local_tokens, args.hidden_size, device=device, dtype=dtype)
    topk_ids = make_topk_ids(
        local_tokens,
        args.topk,
        args.num_experts,
        world_size,
        rank,
        args.routing,
        device,
    ).to(nixl_ep.topk_idx_t)
    topk_weights = torch.rand(
        local_tokens, args.topk, device=device, dtype=torch.float32
    )
    output = torch.empty_like(x)
    combine_sig = inspect.signature(nixl_ep.Buffer.combine)
    combine_supports_out = "out" in combine_sig.parameters
    combine_supports_zero_copy = "zero_copy" in combine_sig.parameters
    payload_bytes = estimate_nixl_remote_payload_bytes(
        topk_ids.long(), args.hidden_size, dtype, rank, world_size, args.num_experts
    )

    buffer = None
    expert_x = None
    handle = None
    last_err: RuntimeError | None = None
    factors = (1,) if args.size_mode == "equal" else NIXL_RDMA_SIZE_SAFETY_FACTORS
    try:
        for factor in factors:
            try:
                buffer = create_nixl_buffer(
                    args,
                    rank,
                    world_size,
                    max_tokens,
                    rdma_safety_factor=factor,
                )
                barrier()
                expert_x, _, handle, _, _ = buffer.dispatch(
                    x,
                    topk_ids,
                    max_tokens,
                    args.num_experts,
                    use_fp8=False,
                    async_finish=False,
                    return_recv_hook=False,
                )
                barrier()
                break
            except RuntimeError as err:
                last_err = err
                err_msg = str(err)
                if buffer is not None:
                    try:
                        buffer.destroy()
                    except Exception:
                        pass
                    buffer = None
                if "layout.total_bytes <= num_rdma_bytes" in err_msg:
                    continue
                raise
        if buffer is None or expert_x is None or handle is None:
            assert last_err is not None
            raise last_err

        fused_out = expert_x.clone()

        def dispatch() -> None:
            assert buffer is not None
            buffer.dispatch(
                x,
                topk_ids,
                max_tokens,
                args.num_experts,
                use_fp8=False,
                async_finish=False,
                return_recv_hook=False,
            )

        def combine() -> None:
            assert buffer is not None and handle is not None
            combine_kwargs: dict[str, Any] = dict(
                async_finish=False,
                return_recv_hook=False,
            )
            if combine_supports_zero_copy:
                combine_kwargs["zero_copy"] = False
            if combine_supports_out:
                combine_kwargs["out"] = output
            buffer.combine(
                fused_out,
                topk_ids,
                topk_weights,
                handle,
                **combine_kwargs,
            )

        if args.mode == "dispatch":
            fn = dispatch
        elif args.mode == "combine":
            fn = combine
        else:

            def fn() -> None:
                assert buffer is not None
                _, _, local_handle, _, _ = buffer.dispatch(
                    x,
                    topk_ids,
                    max_tokens,
                    args.num_experts,
                    use_fp8=False,
                    async_finish=False,
                    return_recv_hook=False,
                )
                combine_kwargs: dict[str, Any] = dict(
                    async_finish=False,
                    return_recv_hook=False,
                )
                if combine_supports_zero_copy:
                    combine_kwargs["zero_copy"] = False
                if combine_supports_out:
                    combine_kwargs["out"] = output
                buffer.combine(
                    fused_out,
                    topk_ids,
                    topk_weights,
                    local_handle,
                    **combine_kwargs,
                )

            payload_bytes *= 2

        stats = run_timed_loop(
            "nixl_ep", args.mode, fn, args, rank, world_size, payload_bytes
        )
        barrier()
        return stats
    finally:
        try:
            buffer.destroy()
        finally:
            barrier()


def run_timed_loop(
    backend: str,
    mode: str,
    fn,
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    payload_bytes: int | None,
) -> BenchStats:
    for _ in range(args.warmup_iters):
        fn()
    barrier()

    lat_us: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(args.iters):
        barrier()
        start.record()
        fn()
        end.record()
        end.synchronize()
        lat_us.append(start.elapsed_time(end) * 1000.0)

    return collect_stats(
        backend=backend,
        mode=mode,
        rank=rank,
        world_size=world_size,
        args=args,
        lat_us=lat_us,
        payload_bytes=payload_bytes,
    )


def print_stats(stats: BenchStats, json_only: bool) -> None:
    if json_only:
        print(json.dumps(asdict(stats), sort_keys=True), flush=True)
        return

    payload_mb = (
        None
        if stats.estimated_payload_bytes_per_rank is None
        else stats.estimated_payload_bytes_per_rank / (1024 * 1024)
    )
    msg = (
        f"[{stats.backend}:{stats.mode}] "
        f"avg={stats.avg_us:.2f}us "
        f"p50={stats.p50_us:.2f}us "
        f"p95={stats.p95_us:.2f}us "
        f"min={stats.min_us:.2f}us "
        f"max={stats.max_us:.2f}us"
    )
    if stats.estimated_bandwidth_gbps is not None and payload_mb is not None:
        msg += (
            f" payload_per_rank={payload_mb:.2f}MiB"
            f" est_bw={stats.estimated_bandwidth_gbps:.2f}GB/s"
        )
    print(msg, flush=True)


def write_csv_rows(path: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def stats_to_row(stats: BenchStats, routing: str) -> dict[str, Any]:
    row = asdict(stats)
    row["routing"] = routing
    return row


def main() -> None:
    args = parse_args()
    rank_env, _, local_rank = get_rank_env()
    init_distributed(local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert rank == rank_env

    device = torch.device("cuda", torch.cuda.current_device())
    dtype = dtype_from_name(args.dtype)

    if args.backend == "both":
        backends = ["agrs", "nixl_ep"]
    elif args.backend == "all":
        backends = ["agrs", "agrs_torch", "nixl_ep"]
    else:
        backends = [args.backend]
    results: list[BenchStats] = []
    csv_rows: list[dict[str, Any]] = []

    try:
        cases = iter_sweep_cases(args) if args.sweep else [args]
        for idx, case_args in enumerate(cases, start=1):
            valid, reason = is_valid_case(case_args, world_size)
            if not valid:
                if rank == 0 and not args.json:
                    print(
                        f"[skip {idx}/{len(cases)}] "
                        f"tokens={case_args.num_tokens} hidden={case_args.hidden_size} "
                        f"experts={case_args.num_experts} topk={case_args.topk}: {reason}",
                        flush=True,
                    )
                continue

            if rank == 0 and args.sweep and not args.json:
                print(
                    f"[case {idx}/{len(cases)}] "
                    f"tokens={case_args.num_tokens} hidden={case_args.hidden_size} "
                    f"experts={case_args.num_experts} topk={case_args.topk}",
                    flush=True,
                )

            for backend in backends:
                valid_backend, backend_reason = is_valid_case_for_backend(
                    case_args, world_size, backend
                )
                if not valid_backend:
                    if rank == 0 and not args.json:
                        print(
                            f"[skip {backend} {idx}/{len(cases)}] "
                            f"tokens={case_args.num_tokens} hidden={case_args.hidden_size} "
                            f"experts={case_args.num_experts} topk={case_args.topk}: "
                            f"{backend_reason}",
                            flush=True,
                        )
                    continue
                barrier()
                if backend == "agrs":
                    stats = benchmark_agrs_pynccl(
                        case_args, rank, world_size, device, dtype
                    )
                elif backend == "agrs_torch":
                    stats = benchmark_agrs_torch(
                        case_args, rank, world_size, device, dtype
                    )
                else:
                    stats = benchmark_nixl_ep(
                        case_args, rank, world_size, device, dtype
                    )
                results.append(stats)
                if rank == 0:
                    csv_rows.append(stats_to_row(stats, case_args.routing))
                    if not args.json:
                        print_stats(stats, args.json)

        if rank == 0 and args.csv_out is not None:
            write_csv_rows(args.csv_out, csv_rows)
            if not args.json:
                print(f"[csv] wrote {len(csv_rows)} rows to {args.csv_out}", flush=True)

        if rank == 0 and args.json:
            payload: dict[str, Any] | list[dict[str, Any]]
            if len(results) == 1:
                payload = asdict(results[0])
            else:
                payload = [asdict(item) for item in results]
            print(json.dumps(payload, sort_keys=True), flush=True)
    finally:
        barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
