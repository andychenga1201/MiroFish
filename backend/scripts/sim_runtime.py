"""Shared scheduling and seed-post helpers for the OASIS runners.

Short runs that pass ``max_rounds`` keep the generated multi-day clock and only
execute a prefix of it. With the usual 60-minute rounds that prefix is
hours 0-4, while generated agents are not active until later in the morning,
so every round selects nobody. A truncated window is realigned onto the
earliest active hour only when that window would otherwise contain no eligible
agent. A full run, and any truncated run that already overlaps an active hour,
keeps the original clock.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

# Same default the runners used inline: 08:00-22:59.
DEFAULT_ACTIVE_HOURS: List[int] = list(range(8, 23))

AgentConfig = Mapping[str, Any]
RoundSchedule = Dict[str, Any]


def prepare_round_schedule(
    total_hours: int,
    minutes_per_round: int,
    max_rounds: int | None,
    agent_configs: Sequence[AgentConfig] | None,
) -> RoundSchedule:
    """Resolve the executed round count and any short-window hour offset."""
    if minutes_per_round is None or minutes_per_round <= 0:
        raise ValueError("minutes_per_round must be positive")

    original_rounds = (total_hours * 60) // minutes_per_round
    total_rounds = original_rounds
    truncated = False
    if max_rounds is not None and max_rounds > 0 and max_rounds < original_rounds:
        truncated = True
        total_rounds = max_rounds

    hour_offset = short_window_hour_offset(
        agent_configs or [],
        total_rounds,
        minutes_per_round,
        truncated=truncated,
    )
    return {
        "total_hours": total_hours,
        "minutes_per_round": minutes_per_round,
        "original_rounds": original_rounds,
        "total_rounds": total_rounds,
        "truncated": truncated,
        "hour_offset": hour_offset,
    }


def round_clock(
    round_num: int,
    minutes_per_round: int,
    hour_offset: int = 0,
) -> Tuple[int, int, int]:
    """Return ``(simulated_minutes, hour_of_day, day_number)`` for one round.

    ``day_number`` stays anchored to elapsed simulated minutes so a realigned
    short window does not pretend to be a later calendar day. Only the hour
    used for activity selection moves.
    """
    simulated_minutes = round_num * minutes_per_round
    simulated_hour = (simulated_minutes // 60 + hour_offset) % 24
    simulated_day = simulated_minutes // (60 * 24) + 1
    return simulated_minutes, simulated_hour, simulated_day


def covered_hours(
    total_rounds: int,
    minutes_per_round: int,
    hour_offset: int = 0,
) -> set[int]:
    return {
        round_clock(round_num, minutes_per_round, hour_offset)[1]
        for round_num in range(total_rounds)
    }


def short_window_hour_offset(
    agent_configs: Sequence[AgentConfig],
    total_rounds: int,
    minutes_per_round: int,
    *,
    truncated: bool,
) -> int:
    """Shift a truncated window that misses every active hour.

    Returns 0 for a full schedule, for a truncated schedule that already
    contains an eligible hour, and when no agent publishes an active hour.
    The offset is added to the simulated hour, so round 0 lands on the
    earliest active hour and later rounds keep ``minutes_per_round`` spacing.
    """
    if not truncated or total_rounds <= 0:
        return 0

    active = _union_active_hours(agent_configs)
    if not active:
        return 0

    if covered_hours(total_rounds, minutes_per_round, 0) & active:
        return 0

    earliest = min(active)
    first_hour = round_clock(0, minutes_per_round, 0)[1]
    return (earliest - first_hour) % 24


def candidate_agent_ids(
    agent_configs: Sequence[AgentConfig],
    current_hour: int,
) -> List[Tuple[int, float]]:
    """Agents whose configured active hours include ``current_hour``.

    Activity-level sampling stays in the runners. This is only the clock gate
    that previously rejected every agent during hours 0-4.
    """
    candidates: List[Tuple[int, float]] = []
    for cfg in agent_configs:
        agent_id = cfg.get("agent_id", 0)
        if current_hour not in _normalized_active_hours(cfg):
            continue
        candidates.append((agent_id, cfg.get("activity_level", 0.5)))
    return candidates


def append_initial_post_action(
    actions: MutableMapping[Any, Any],
    agent: Any,
    action: Any,
) -> None:
    """Queue a seed post without collapsing another post from the same agent.

    OASIS ``env.step`` accepts either one action or a list of actions per
    agent. A plain dict assignment keeps only the last post when several
    initial posts share ``poster_agent_id``.
    """
    existing = actions.get(agent)
    if existing is None:
        actions[agent] = action
        return
    if isinstance(existing, list):
        existing.append(action)
        return
    actions[agent] = [existing, action]


def flatten_agent_actions(actions: Mapping[Any, Any]) -> List[Tuple[Any, Any]]:
    """Expand the step payload the same way OASIS walks ``env.step`` actions."""
    flattened: List[Tuple[Any, Any]] = []
    for agent, action in actions.items():
        if isinstance(action, list):
            flattened.extend((agent, item) for item in action)
        else:
            flattened.append((agent, action))
    return flattened


def count_initial_actions(actions: Mapping[Any, Any]) -> int:
    return len(flatten_agent_actions(actions))


def _union_active_hours(agent_configs: Iterable[AgentConfig]) -> set[int]:
    hours: set[int] = set()
    for cfg in agent_configs:
        hours.update(_normalized_active_hours(cfg))
    return hours


def _normalized_active_hours(cfg: AgentConfig) -> List[int]:
    raw = cfg.get("active_hours", DEFAULT_ACTIVE_HOURS)
    hours: List[int] = []
    for hour in raw:
        if isinstance(hour, bool) or not isinstance(hour, int):
            continue
        if 0 <= hour <= 23:
            hours.append(hour)
    return hours
