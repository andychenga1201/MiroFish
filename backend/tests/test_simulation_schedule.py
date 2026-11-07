"""Offline checks for the reported 5x60 short-run schedule.

The sanitized SoaRank run configured 48 hours at 60 minutes per round, then
truncated execution to 5 rounds. Those rounds are hours 0-4. Every agent in
that run is inactive before hour 8, so the activity gate selects nobody and
the loop finishes without actions. Full schedules must still start at 00:00.
"""

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from sim_runtime import (  # noqa: E402
    candidate_agent_ids,
    covered_hours,
    prepare_round_schedule,
    round_clock,
)


def _hours(start: int, end: int) -> list[int]:
    return list(range(start, end + 1))


# Active hours copied from the sanitized simulation config. Post text and
# entity names are not part of this fixture.
REPORTED_ACTIVE_HOURS = {
    0: _hours(9, 16),
    1: _hours(8, 23),
    2: _hours(18, 23),
    3: _hours(9, 16),
    4: _hours(8, 19),
    5: _hours(18, 23),
    6: _hours(8, 23),
    7: _hours(18, 23),
    8: _hours(8, 23),
    9: _hours(8, 23),
    10: _hours(9, 16),
    11: _hours(8, 23),
    12: _hours(8, 23),
    13: _hours(8, 23),
    14: _hours(8, 23),
}


def reported_agent_configs() -> list[dict]:
    return [
        {"agent_id": agent_id, "active_hours": hours, "activity_level": 1.0}
        for agent_id, hours in REPORTED_ACTIVE_HOURS.items()
    ]


def test_reported_config_earliest_active_hour_is_8():
    earliest = min(hour for hours in REPORTED_ACTIVE_HOURS.values() for hour in hours)
    assert earliest == 8
    assert candidate_agent_ids(reported_agent_configs(), 0) == []
    assert candidate_agent_ids(reported_agent_configs(), 7) == []
    assert {agent_id for agent_id, _ in candidate_agent_ids(reported_agent_configs(), 8)} >= {
        1,
        4,
        6,
    }


def test_reported_five_round_truncation_reaches_active_hours():
    schedule = prepare_round_schedule(
        total_hours=48,
        minutes_per_round=60,
        max_rounds=5,
        agent_configs=reported_agent_configs(),
    )

    assert schedule["original_rounds"] == 48
    assert schedule["total_rounds"] == 5
    assert schedule["truncated"] is True
    assert schedule["hour_offset"] == 8
    assert covered_hours(5, 60, 0) == {0, 1, 2, 3, 4}

    hours = []
    for round_num in range(schedule["total_rounds"]):
        _minutes, hour, day = round_clock(
            round_num, schedule["minutes_per_round"], schedule["hour_offset"]
        )
        hours.append(hour)
        assert day == 1
        eligible = candidate_agent_ids(reported_agent_configs(), hour)
        assert eligible, f"round {round_num} hour {hour} still has no eligible agent"
    assert hours == [8, 9, 10, 11, 12]


def test_full_48_hour_schedule_still_starts_at_midnight():
    configs = reported_agent_configs()
    schedule = prepare_round_schedule(
        total_hours=48,
        minutes_per_round=60,
        max_rounds=None,
        agent_configs=configs,
    )

    assert schedule["truncated"] is False
    assert schedule["total_rounds"] == 48
    assert schedule["hour_offset"] == 0
    assert [round_clock(i, 60, 0)[1] for i in range(8)] == list(range(8))
    assert candidate_agent_ids(configs, round_clock(0, 60, 0)[1]) == []
    assert candidate_agent_ids(configs, round_clock(8, 60, 0)[1])


def test_truncation_that_already_overlaps_active_hours_keeps_the_clock():
    configs = reported_agent_configs()
    schedule = prepare_round_schedule(
        total_hours=48,
        minutes_per_round=60,
        max_rounds=9,
        agent_configs=configs,
    )

    assert schedule["truncated"] is True
    assert schedule["hour_offset"] == 0
    assert candidate_agent_ids(configs, round_clock(0, 60, 0)[1]) == []
    assert candidate_agent_ids(configs, round_clock(8, 60, 0)[1])


def test_configured_short_timeline_is_not_realigned():
    """A 5-hour config is the whole timeline, not a truncated prefix."""
    schedule = prepare_round_schedule(
        total_hours=5,
        minutes_per_round=60,
        max_rounds=None,
        agent_configs=reported_agent_configs(),
    )
    assert schedule["truncated"] is False
    assert schedule["total_rounds"] == 5
    assert schedule["hour_offset"] == 0


def test_half_hour_rounds_align_when_the_prefix_is_entirely_inactive():
    schedule = prepare_round_schedule(
        total_hours=48,
        minutes_per_round=30,
        max_rounds=5,
        agent_configs=reported_agent_configs(),
    )
    assert schedule["original_rounds"] == 96
    assert schedule["hour_offset"] == 8
    hours = [
        round_clock(i, 30, schedule["hour_offset"])[1]
        for i in range(5)
    ]
    assert hours == [8, 8, 9, 9, 10]


def test_missing_active_hours_use_the_historical_default_window():
    configs = [{"agent_id": 3}]
    assert candidate_agent_ids(configs, 7) == []
    assert candidate_agent_ids(configs, 8) == [(3, 0.5)]
    assert candidate_agent_ids(configs, 22) == [(3, 0.5)]
    assert candidate_agent_ids(configs, 23) == []


def test_runners_share_the_schedule_helper_and_keep_activity_sampling():
    runners = [
        SCRIPTS / "run_twitter_simulation.py",
        SCRIPTS / "run_reddit_simulation.py",
        SCRIPTS / "run_parallel_simulation.py",
    ]
    for path in runners:
        source = path.read_text(encoding="utf-8")
        assert "prepare_round_schedule(" in source
        assert "round_clock(" in source
        assert "candidate_agent_ids(" in source
        assert "(simulated_minutes // 60) % 24" not in source
        assert "random.uniform(base_min, base_max)" in source
        assert "random.sample(" in source


def test_minutes_per_round_must_be_positive():
    with pytest.raises(ValueError):
        prepare_round_schedule(48, 0, 5, [])
