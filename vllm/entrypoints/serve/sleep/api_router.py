# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import asyncio
import time

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response

import vllm.envs as envs
from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger

logger = init_logger(__name__)


def elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 3)


def optional_tags(payload: dict, key: str) -> list[str] | None:
    tags = payload.get(key)
    if tags is None:
        return None
    if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
        raise HTTPException(status_code=400, detail=f"{key} must be a list of strings")
    if not tags:
        raise HTTPException(status_code=400, detail=f"{key} must not be empty")
    return tags


def required_tags(payload: dict, key: str = "tags") -> list[str]:
    tags = optional_tags(payload, key)
    if tags is None:
        raise HTTPException(status_code=400, detail=f"{key} is required")
    return tags


def optional_bool(payload: dict, key: str, default: bool) -> bool:
    value = payload.get(key, default)
    if not isinstance(value, bool):
        raise HTTPException(status_code=400, detail=f"{key} must be a boolean")
    return value


def optional_timeout(payload: dict, key: str, default: float) -> float:
    value = payload.get(key, default)
    if not isinstance(value, (int, float)) or value <= 0:
        raise HTTPException(status_code=400, detail=f"{key} must be a positive number")
    return float(value)

def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


def flash_epscale_lock(request: Request) -> asyncio.Lock:
    lock = getattr(request.app.state, "flash_epscale_lock", None)
    if lock is None:
        lock = asyncio.Lock()
        request.app.state.flash_epscale_lock = lock
    return lock


async def _get_consistent_ep_sleep_state(client: EngineClient) -> dict[str, object]:
    states = await client.collective_rpc("get_ep_sleep_state")
    if not states:
        raise HTTPException(status_code=500, detail="failed to query EP sleep state")

    first_state = states[0]
    if any(state != first_state for state in states[1:]):
        raise HTTPException(
            status_code=500,
            detail=f"inconsistent EP sleep state across workers: {states}",
        )

    return first_state


router = APIRouter()


@router.post("/sleep")
async def sleep(raw_request: Request):
    # get POST params
    level = raw_request.query_params.get("level", "1")
    mode = raw_request.query_params.get("mode", "abort")
    await engine_client(raw_request).sleep(int(level), mode)
    # FIXME: in v0 with frontend multiprocessing, the sleep command
    # is sent but does not finish yet when we return a response.
    return Response(status_code=200)


@router.post("/wake_up")
async def wake_up(raw_request: Request):
    tags = raw_request.query_params.getlist("tags")
    if tags == []:
        # set to None to wake up all tags if no tags are provided
        tags = None
    logger.info("wake up the engine with tags: %s", tags)
    await engine_client(raw_request).wake_up(tags)
    # FIXME: in v0 with frontend multiprocessing, the wake-up command
    # is sent but does not finish yet when we return a response.
    return Response(status_code=200)


@router.get("/is_sleeping")
async def is_sleeping(raw_request: Request):
    is_sleeping = await engine_client(raw_request).is_sleeping()
    return JSONResponse(content={"is_sleeping": is_sleeping})



@router.post("/sleep_ep_ranks_tags")
async def sleep_ep_ranks_by_tags(raw_request: Request):
    payload = await raw_request.json()
    sleeping_ep_ranks = payload["sleeping_ep_ranks"]
    tags = required_tags(payload, "tags")

    await engine_client(raw_request).collective_rpc(
        "sleep_ep_ranks_by_tags",
        kwargs={
            "sleeping_ep_ranks": sleeping_ep_ranks,
            "tags": tags,
        },
    )
    return JSONResponse(
        content={
            "ok": True,
            "sleeping_ep_ranks": sleeping_ep_ranks,
            "tags": tags,
        }
    )


@router.post("/wake_up_ep_ranks_tags")
async def wake_up_ep_ranks_by_tags(raw_request: Request):
    payload = await raw_request.json()
    sleeping_ep_ranks = payload["sleeping_ep_ranks"]
    tags = required_tags(payload, "tags")

    await engine_client(raw_request).collective_rpc(
        "wake_up_ep_ranks",
        kwargs={
            "sleeping_ep_ranks": sleeping_ep_ranks,
            "tags": tags,
        },
    )
    return JSONResponse(
        content={
            "ok": True,
            "sleeping_ep_ranks": sleeping_ep_ranks,
            "tags": tags,
        }
    )

@router.post("/flash_epscale")
async def wscale(raw_request: Request):
    payload = await raw_request.json()
    target_ep_size = payload.get("ep_size")
    if not isinstance(target_ep_size, int):
        raise HTTPException(status_code=400, detail="ep_size must be an integer")
    #tags = optional_tags(payload, "tags") or ["expert_weights","shared_weights","kv_cache"]
    tags = optional_tags(payload, "tags") or ["expert_weights","kv_cache"]
    client = engine_client(raw_request)
    drain_timeout = optional_timeout(payload, "drain_timeout", 300)
    set_active_dp_size = getattr(client, "set_active_data_parallel_size", None)
    wait_for_dp_drain = getattr(client, "wait_for_dp_ranks_to_drain", None)
    pause_generation = getattr(client, "pause_generation", None)
    resume_generation = getattr(client, "resume_generation", None)
    timing_ms: dict[str, float] = {}
    total_start = time.perf_counter()

    async with flash_epscale_lock(raw_request):
        step_start = time.perf_counter()
        first_state = await _get_consistent_ep_sleep_state(client)
        timing_ms["query_state"] = elapsed_ms(step_start)
        ep_world_size = int(first_state["ep_world_size"])
        active_ep_size = int(first_state["active_ep_size"])
        current_sleeping = [int(rank) for rank in first_state["sleeping_ep_ranks"]]

        if target_ep_size <= 0 or target_ep_size > ep_world_size:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"ep_size must be in [1, {ep_world_size}], "
                    f"got {target_ep_size}"
                ),
            )

        if target_ep_size == active_ep_size:
            step_start = time.perf_counter()
            if set_active_dp_size is not None:
                set_active_dp_size(target_ep_size)
            timing_ms["route"] = elapsed_ms(step_start)
            timing_ms["total"] = elapsed_ms(total_start)
            logger.info("flash_epscale noop timing_ms=%s", timing_ms)
            return JSONResponse(
                content={
                    "ok": True,
                    "ep_world_size": ep_world_size,
                    "active_ep_size": active_ep_size,
                    "sleeping_ep_ranks": current_sleeping,
                    "changed": False,
                    "action": "noop",
                    "tags": tags,
                    "timing_ms": timing_ms,
                }
            )

        target_sleeping = list(range(target_ep_size, ep_world_size))
        paused = False

        if target_ep_size < active_ep_size:
            action = "scale_down"
            try:
                step_start = time.perf_counter()
                if set_active_dp_size is not None:
                    set_active_dp_size(target_ep_size)
                timing_ms["route_shrink"] = elapsed_ms(step_start)
                if wait_for_dp_drain is not None:
                    step_start = time.perf_counter()
                    await wait_for_dp_drain(target_sleeping, drain_timeout)
                    timing_ms["drain"] = elapsed_ms(step_start)
            except Exception as e:
                step_start = time.perf_counter()
                if set_active_dp_size is not None:
                    set_active_dp_size(active_ep_size)
                timing_ms["route_restore"] = elapsed_ms(step_start)
                timing_ms["total"] = elapsed_ms(total_start)
                logger.exception(
                    "flash_epscale scale_down drain failed timing_ms=%s",
                    timing_ms,
                )
                raise HTTPException(
                    status_code=500,
                    detail=f"flash_epscale scale_down drain failed: {e}",
                ) from e

            try:
                if pause_generation is not None:
                    step_start = time.perf_counter()
                    await pause_generation(mode="wait", clear_cache=False)
                    paused = True
                    timing_ms["pause"] = elapsed_ms(step_start)
                if current_sleeping:
                    step_start = time.perf_counter()
                    await client.collective_rpc(
                        "wake_up_ep_ranks",
                        kwargs={
                            "sleeping_ep_ranks": current_sleeping,
                            "tags": tags,
                        },
                    )
                    timing_ms["wake"] = elapsed_ms(step_start)
                step_start = time.perf_counter()
                await client.collective_rpc(
                    "resize_sleep_ep_ranks",
                    kwargs={"sleeping_ep_ranks": target_sleeping},
                )
                timing_ms["resize"] = elapsed_ms(step_start)
                step_start = time.perf_counter()
                await client.collective_rpc(
                    "sleep_ep_ranks_by_tags",
                    kwargs={
                        "sleeping_ep_ranks": target_sleeping,
                        "tags": tags,
                    },
                )
                timing_ms["sleep"] = elapsed_ms(step_start)
            except Exception as e:
                timing_ms["total"] = elapsed_ms(total_start)
                logger.exception("flash_epscale scale_down failed")
                if paused and resume_generation is not None:
                    try:
                        step_start = time.perf_counter()
                        await resume_generation()
                        timing_ms["resume_after_error"] = elapsed_ms(step_start)
                    except Exception:
                        logger.exception("flash_epscale scale_down resume failed")
                logger.error(
                    "flash_epscale scale_down failed timing_ms=%s", timing_ms
                )
                raise HTTPException(
                    status_code=500,
                    detail=f"flash_epscale scale_down failed: {e}",
                ) from e
        else:
            action = "scale_up"
            try:
                if pause_generation is not None:
                    step_start = time.perf_counter()
                    await pause_generation(mode="wait", clear_cache=False)
                    paused = True
                    timing_ms["pause"] = elapsed_ms(step_start)
                if current_sleeping:
                    step_start = time.perf_counter()
                    await client.collective_rpc(
                        "wake_up_ep_ranks",
                        kwargs={
                            "sleeping_ep_ranks": current_sleeping,
                            "tags": tags,
                        },
                    )
                    timing_ms["wake"] = elapsed_ms(step_start)
                step_start = time.perf_counter()
                await client.collective_rpc(
                    "resize_sleep_ep_ranks",
                    kwargs={"sleeping_ep_ranks": target_sleeping},
                )
                timing_ms["resize"] = elapsed_ms(step_start)
                if target_sleeping:
                    step_start = time.perf_counter()
                    await client.collective_rpc(
                        "sleep_ep_ranks_by_tags",
                        kwargs={
                            "sleeping_ep_ranks": target_sleeping,
                            "tags": tags,
                        },
                    )
                    timing_ms["sleep"] = elapsed_ms(step_start)
            except Exception as e:
                if current_sleeping:
                    try:
                        await client.collective_rpc(
                            "sleep_ep_ranks_by_tags",
                            kwargs={
                                "sleeping_ep_ranks": current_sleeping,
                                "tags": tags,
                            },
                        )
                    except Exception:
                        logger.exception("flash_epscale scale_up rollback sleep failed")
                if paused and resume_generation is not None:
                    try:
                        step_start = time.perf_counter()
                        await resume_generation()
                        timing_ms["resume_after_error"] = elapsed_ms(step_start)
                    except Exception:
                        logger.exception("flash_epscale scale_up resume failed")
                timing_ms["total"] = elapsed_ms(total_start)
                logger.exception(
                    "flash_epscale scale_up failed timing_ms=%s", timing_ms
                )
                raise HTTPException(
                    status_code=500,
                    detail=f"flash_epscale scale_up failed: {e}",
                ) from e

        step_start = time.perf_counter()
        final_state = await _get_consistent_ep_sleep_state(client)
        timing_ms["final_state"] = elapsed_ms(step_start)
        final_active_ep_size = int(final_state["active_ep_size"])
        final_sleeping = [int(rank) for rank in final_state["sleeping_ep_ranks"]]
        if final_active_ep_size != target_ep_size or final_sleeping != target_sleeping:
            if paused and resume_generation is not None:
                try:
                    step_start = time.perf_counter()
                    await resume_generation()
                    timing_ms["resume_after_error"] = elapsed_ms(step_start)
                except Exception:
                    logger.exception("flash_epscale final-state resume failed")
            timing_ms["total"] = elapsed_ms(total_start)
            logger.error(
                "flash_epscale final state mismatch timing_ms=%s", timing_ms
            )
            raise HTTPException(
                status_code=500,
                detail=(
                    "flash_epscale finished with unexpected EP sleep state: "
                    f"expected active_ep_size={target_ep_size}, "
                    f"sleeping_ep_ranks={target_sleeping}, got "
                    f"active_ep_size={final_active_ep_size}, "
                    f"sleeping_ep_ranks={final_sleeping}"
                ),
            )
        step_start = time.perf_counter()
        if set_active_dp_size is not None:
            set_active_dp_size(target_ep_size)
        timing_ms["route_final"] = elapsed_ms(step_start)
        if paused and resume_generation is not None:
            step_start = time.perf_counter()
            await resume_generation()
            timing_ms["resume"] = elapsed_ms(step_start)
        timing_ms["total"] = elapsed_ms(total_start)
        logger.info("flash_epscale %s timing_ms=%s", action, timing_ms)

        return JSONResponse(
            content={
                "ok": True,
                "ep_world_size": ep_world_size,
                "active_ep_size": final_active_ep_size,
                "sleeping_ep_ranks": final_sleeping,
                "changed": True,
                "action": action,
                "tags": tags,
                "timing_ms": timing_ms,
            }
        )


def attach_router(app: FastAPI):
    if not envs.VLLM_SERVER_DEV_MODE:
        return

    app.include_router(router)
