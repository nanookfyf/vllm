# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response

import vllm.envs as envs
from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger

logger = init_logger(__name__)



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

def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


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
    tags = optional_tags(payload, "tags") or ["expert_weights"]

    client = engine_client(raw_request)
    first_state = await _get_consistent_ep_sleep_state(client)
    ep_world_size = int(first_state["ep_world_size"])
    active_ep_size = int(first_state["active_ep_size"])
    current_sleeping = [int(rank) for rank in first_state["sleeping_ep_ranks"]]

    if target_ep_size <= 0 or target_ep_size > ep_world_size:
        raise HTTPException(
            status_code=400,
            detail=f"ep_size must be in [1, {ep_world_size}], got {target_ep_size}",
        )

    if target_ep_size == active_ep_size:
        return JSONResponse(
            content={
                "ok": True,
                "ep_world_size": ep_world_size,
                "active_ep_size": active_ep_size,
                "sleeping_ep_ranks": current_sleeping,
                "changed": False,
                "action": "noop",
                "tags": tags,
            }
        )

    target_sleeping = list(range(target_ep_size, ep_world_size))

    if target_ep_size < active_ep_size:
        newly_sleeping = [rank for rank in target_sleeping if rank not in current_sleeping]
        try:
            await client.collective_rpc(
                "resize_sleep_ep_ranks",
                kwargs={"sleeping_ep_ranks": target_sleeping},
            )
            if newly_sleeping:
                await client.collective_rpc(
                    "sleep_ep_ranks_by_tags",
                    kwargs={
                        "sleeping_ep_ranks": newly_sleeping,
                        "tags": tags,
                    },
                )
        except Exception as e:
            try:
                await client.collective_rpc(
                    "resize_sleep_ep_ranks",
                    kwargs={"sleeping_ep_ranks": current_sleeping},
                )
            except Exception:
                logger.exception("flash_epscale scale_down rollback failed")
            raise HTTPException(
                status_code=500,
                detail=f"flash_epscale scale_down failed: {e}",
            ) from e
        action = "scale_down"
    else:
        waking_ranks = [rank for rank in current_sleeping if rank not in target_sleeping]
        try:
            if waking_ranks:
                await client.collective_rpc(
                    "wake_up_ep_ranks",
                    kwargs={
                        "sleeping_ep_ranks": waking_ranks,
                        "tags": tags,
                    },
                )
            await client.collective_rpc(
                "resize_sleep_ep_ranks",
                kwargs={"sleeping_ep_ranks": target_sleeping},
            )
        except Exception as e:
            if waking_ranks:
                try:
                    await client.collective_rpc(
                        "sleep_ep_ranks_by_tags",
                        kwargs={
                            "sleeping_ep_ranks": waking_ranks,
                            "tags": tags,
                        },
                    )
                except Exception:
                    logger.exception("flash_epscale scale_up rollback sleep failed")
            try:
                await client.collective_rpc(
                    "resize_sleep_ep_ranks",
                    kwargs={"sleeping_ep_ranks": current_sleeping},
                )
            except Exception:
                logger.exception("flash_epscale scale_up rollback resize failed")
            raise HTTPException(
                status_code=500,
                detail=f"flash_epscale scale_up failed: {e}",
            ) from e

        action = "scale_up"

    final_state = await _get_consistent_ep_sleep_state(client)
    final_active_ep_size = int(final_state["active_ep_size"])
    final_sleeping = [int(rank) for rank in final_state["sleeping_ep_ranks"]]
    if final_active_ep_size != target_ep_size or final_sleeping != target_sleeping:
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

    return JSONResponse(
        content={
            "ok": True,
            "ep_world_size": ep_world_size,
            "active_ep_size": final_active_ep_size,
            "sleeping_ep_ranks": final_sleeping,
            "changed": True,
            "action": action,
            "tags": tags,
        }
    )


def attach_router(app: FastAPI):
    if not envs.VLLM_SERVER_DEV_MODE:
        return

    app.include_router(router)
