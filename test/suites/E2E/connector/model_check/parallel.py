# SPDX-License-Identifier: MIT
"""Single-node Gloo launch, native TP/PP/PCP/DCP rank discovery and handshakes."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RankCoords:
    """Native distributed coordinates reported by vLLM process groups."""

    world_rank: int
    pp_rank: int
    pcp_rank: int
    tp_rank: int
    dcp_rank: int
    pp_size: int
    pcp_size: int
    tp_size: int
    dcp_size: int


def launch(args) -> bool:
    """Supervise all TP x PCP x PP worker ranks with single-node torchrun."""
    world_size = args.tp * args.pcp * args.pp
    if "RANK" in os.environ:
        if (
            int(os.environ["WORLD_SIZE"]) != world_size
            or int(os.environ["LOCAL_WORLD_SIZE"]) != world_size
            or int(os.environ["LOCAL_RANK"]) != int(os.environ["RANK"])
        ):
            raise ValueError(
                f"Expected a single-node torchrun with TP*PCP*PP={world_size} processes"
            )
        return False
    if world_size == 1:
        return False
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={world_size}",
            str(Path(__file__).with_name("run.py")),
            *sys.argv[1:],
        ],
        check=True,
    )
    return True


def gather(value):
    import torch.distributed as dist

    world_size = dist.get_world_size()
    if world_size == 1:
        return [value]
    values = [None] * world_size
    dist.all_gather_object(values, value)
    return values


def broadcast(value):
    import torch.distributed as dist

    if dist.get_world_size() == 1:
        return value
    items = [value]
    dist.broadcast_object_list(items, src=0)
    return items[0]


def barrier() -> None:
    import torch.distributed as dist

    if dist.get_world_size() > 1:
        dist.barrier()


def expected_rank_coords(
    world_rank: int, tp_size: int, pcp_size: int, pp_size: int
) -> tuple[int, int, int]:
    """Return (pp_rank, pcp_rank, tp_rank) matching vLLM's PP x PCP x TP grid."""
    stage_width = tp_size * pcp_size
    pp_rank, within_stage = divmod(world_rank, stage_width)
    pcp_rank, tp_rank = divmod(within_stage, tp_size)
    if not 0 <= pp_rank < pp_size:
        raise AssertionError(f"World rank {world_rank} out of bounds for PP={pp_size}")
    return pp_rank, pcp_rank, tp_rank


def inspect_local_coords(args) -> RankCoords:
    """Read and verify this worker's native PP/PCP/TP/DCP group coordinates."""
    import torch.distributed as dist
    from vllm.distributed.parallel_state import (
        get_dcp_group,
        get_pcp_group,
        get_pp_group,
        get_tp_group,
    )

    world_rank = dist.get_rank()
    pp_group = get_pp_group()
    pcp_group = get_pcp_group()
    tp_group = get_tp_group()
    dcp_group = get_dcp_group()

    actual_sizes = (
        tp_group.world_size,
        pp_group.world_size,
        pcp_group.world_size,
        dcp_group.world_size,
    )
    requested_sizes = (args.tp, args.pp, args.pcp, args.dcp)
    if actual_sizes != requested_sizes:
        raise AssertionError(
            f"Native group sizes (TP, PP, PCP, DCP)={actual_sizes} "
            f"disagree with requested {requested_sizes}"
        )

    expected_pp, expected_pcp, expected_tp = expected_rank_coords(
        world_rank, args.tp, args.pcp, args.pp
    )
    actual_tuple = (
        pp_group.rank_in_group,
        pcp_group.rank_in_group,
        tp_group.rank_in_group,
    )
    if actual_tuple != (expected_pp, expected_pcp, expected_tp):
        raise AssertionError(
            f"Rank {world_rank} coordinates (pp, pcp, tp)={actual_tuple} "
            f"disagree with expected {(expected_pp, expected_pcp, expected_tp)}"
        )

    return RankCoords(
        world_rank=world_rank,
        pp_rank=pp_group.rank_in_group,
        pcp_rank=pcp_group.rank_in_group,
        tp_rank=tp_group.rank_in_group,
        dcp_rank=dcp_group.rank_in_group,
        pp_size=pp_group.world_size,
        pcp_size=pcp_group.world_size,
        tp_size=tp_group.world_size,
        dcp_size=dcp_group.world_size,
    )


def collect_handshakes(worker, coords: RankCoords, args) -> dict:
    """Gather connector handshake metadata from every rank in vLLM key format.

    vLLM publishes either (pp_rank, pcp_rank, tp_rank) when PCP > 1 or
    (pp_rank, tp_rank) when PCP == 1. Both v1 PP/CP connectors and v2 accept
    these keys via set_xfer_handshake_metadata_pp_aware().
    """
    local_handshake = worker.get_handshake_metadata()
    reports = gather((coords, local_handshake))
    handshakes = {}
    for report_coords, metadata in reports:
        if metadata is None:
            continue
        if args.pcp > 1:
            key = (report_coords.pp_rank, report_coords.pcp_rank, report_coords.tp_rank)
        else:
            key = (report_coords.pp_rank, report_coords.tp_rank)
        handshakes[key] = metadata
    return handshakes
