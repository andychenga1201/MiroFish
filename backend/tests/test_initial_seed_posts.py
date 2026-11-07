"""Same-author initial posts must all be queued for OASIS env.step."""

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from sim_runtime import (  # noqa: E402
    append_initial_post_action,
    count_initial_actions,
    flatten_agent_actions,
)


class _Agent:
    def __init__(self, agent_id: int):
        self.agent_id = agent_id


def _action(content: str) -> dict:
    return {"action_type": "CREATE_POST", "content": content}


def test_repeated_poster_keeps_every_seed_post_in_order():
    author = _Agent(10)
    actions = {}
    contents = [f"seed-{index}" for index in range(5)]
    for content in contents:
        append_initial_post_action(actions, author, _action(content))

    flattened = flatten_agent_actions(actions)
    assert count_initial_actions(actions) == 5
    assert len(actions) == 1
    assert [action["content"] for _agent, action in flattened] == contents
    assert all(agent is author for agent, _action in flattened)


def test_distinct_authors_stay_single_actions():
    first = _Agent(1)
    second = _Agent(2)
    actions = {}
    append_initial_post_action(actions, first, _action("one"))
    append_initial_post_action(actions, second, _action("two"))

    assert count_initial_actions(actions) == 2
    assert actions[first] == _action("one")
    assert not isinstance(actions[first], list)
    assert actions[second] == _action("two")


def test_mixed_authors_group_only_the_repeated_poster():
    repeated = _Agent(10)
    other = _Agent(4)
    actions = {}
    append_initial_post_action(actions, repeated, _action("a"))
    append_initial_post_action(actions, other, _action("b"))
    append_initial_post_action(actions, repeated, _action("c"))

    flattened = flatten_agent_actions(actions)
    assert [action["content"] for _agent, action in flattened] == ["a", "c", "b"]
    assert isinstance(actions[repeated], list)
    assert actions[other] == _action("b")


def test_runners_do_not_collapse_seed_posts_by_dict_assignment():
    runners = [
        SCRIPTS / "run_twitter_simulation.py",
        SCRIPTS / "run_reddit_simulation.py",
        SCRIPTS / "run_parallel_simulation.py",
    ]
    for path in runners:
        source = path.read_text(encoding="utf-8")
        assert "append_initial_post_action(" in source
        assert "count_initial_actions(" in source
        assert "initial_actions[agent] =" not in source
