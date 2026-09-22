"""Per-turn tool-call reward shaping in TrajectoryManager.

Drives TrajectoryManager with an injected fake turn_scorer (no annotator
dependency) and asserts the dense per-token shaping vector written to
Sample.metadata["toolcall_turn_shaping"] is correctly valued, aligned to
loss_mask, and bounded by the per-trajectory budget.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from slime.agent.adapters.common import TurnRecord  # noqa: E402
from slime.agent.trajectory import TrajectoryManager  # noqa: E402
from slime.utils.types import Sample  # noqa: E402

# Token bands: system=1000, user=2000, assistant=9000, tool=3000
SYS = [1000, 1001, 1099]
USR = [2000, 2001, 2099]


def _asst_msg(label):
    return {"role": "assistant", "content": label}


def _user_msg(ids):
    return {"role": "user", "content": str(ids)}


def _turn(prompt_ids, response_ids):
    return TurnRecord(
        prompt_ids=list(prompt_ids),
        output_ids=list(response_ids),
        finish_reason="stop",
        output_log_probs=[0.0] * len(response_ids),
    )


def _two_turn_session(mgr, sid, *, r1, r2):
    """Two clean, prefix-extending turns. Returns (prompt2_len,)."""
    p1 = SYS + USR
    resp1 = [9001, 9002, 9003]  # 3 response tokens
    mgr.record_turn(
        sid,
        turn=_turn(p1, resp1),
        prompt_messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
        response_message=_asst_msg(r1),
    )
    p2 = p1 + resp1 + [2100, 2101]  # prior + tool/user follow-up
    resp2 = [9004, 9005]  # 2 response tokens
    mgr.record_turn(
        sid,
        turn=_turn(p2, resp2),
        prompt_messages=[
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            _asst_msg(r1),
            {"role": "user", "content": "f"},
        ],
        response_message=_asst_msg(r2),
    )


def test_realign_masked_span_excluded_from_budget():
    """REALIGN masks out turn 1's response (loss_mask=0), but its turn_spans entry
    (trained=True) survives. Shaping must NOT spend budget on those masked-out
    tokens: the shaping vector must be 0 wherever loss_mask==0, and a 1.0 budget
    must land entirely on the live (loss_mask==1) tokens.

    Construction mirrors test_2_4_drift_case_B1_short_replaces in
    test_trajectory_manager_branching.py: turn 2's prompt drifts inside turn 1's
    most-recent response span (drift_replace at the last echoed token) and the
    incoming response is short (< fork_threshold), so classify_token_drift returns
    REALIGN — overwriting turn 1's response as loss_mask=0.
    """

    # scorer flags turn 1 (the realigned/masked span) as errored.
    def scorer(node):
        return 1

    mgr = TrajectoryManager(turn_scorer=scorer, shaping_beta=0.25, shaping_budget=1.0)
    sid = "realign"

    # turn 1: system+user prompt, 3-token trained response.
    p1 = SYS + USR + [9000]  # prompt ends with the assistant/gen marker
    resp1 = [9001, 9002, 9003]
    mgr.record_turn(
        sid,
        turn=_turn(p1, resp1),
        prompt_messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
        response_message=_asst_msg("a1"),
    )

    # turn 2: prompt echoes p1 + resp1, then a tool/user follow-up + gen marker,
    # but with a drift INSIDE resp1's echoed span (last echoed token replaced).
    p2_honest = p1 + resp1 + [3000, 3001, 9000]
    drift_idx = len(p1) + len(resp1) - 1  # last token of resp1's echo -> inside the span
    p2 = list(p2_honest)
    p2[drift_idx] = 7001  # sentinel drift token (drift_replace)
    resp2 = [9004, 9005]  # short -> len < default fork_threshold -> REALIGN
    mgr.record_turn(
        sid,
        turn=_turn(p2, resp2),
        prompt_messages=[
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            _asst_msg("a1"),
            {"role": "user", "content": "f"},
        ],
        response_message=_asst_msg("a2"),
    )

    samples = mgr.get_trajectory(sid, base_sample=Sample(index=0, prompt=""), reward=1.0)
    assert len(samples) == 1, f"REALIGN should yield one merged sample, got {len(samples)}"
    s = samples[0]
    vec = s.metadata["toolcall_turn_shaping"]
    assert len(vec) == s.response_length == len(s.loss_mask)

    # (1) Shaping is zero at every masked-out position (the realigned turn-1 span).
    for v, m in zip(vec, s.loss_mask, strict=True):
        if m == 0:
            assert v == 0.0, "shaping leaked onto a loss_mask==0 token"

    # (2) There is at least one masked-out token (proving REALIGN actually fired
    # and turn 1's trained span was demoted to context).
    assert 0 in s.loss_mask, "expected REALIGN to demote turn-1 response to loss_mask=0"

    # (3) Turn 2 has 1 error and 2 live tokens. The turn's TOTAL penalty is a
    # fixed -beta*errors = -0.25, distributed uniformly over its 2 live tokens ->
    # -0.125 each; sum = -0.25, well under the 1.0 budget, so no scale-down.
    # Crucially the 3 masked-out turn-1 tokens contribute 0 to total_abs (both to
    # the budget denominator and the shaping), so total_abs == 0.25 exactly.
    total_abs = sum(abs(v) for v in vec)
    assert abs(total_abs - 0.25) < 1e-9, f"budget denominator included masked tokens: {total_abs}"
    live_nonzero = [v for v, m in zip(vec, s.loss_mask, strict=True) if m == 1 and v != 0.0]
    assert live_nonzero == [-0.125, -0.125], f"live tokens not penalized as fixed/-N: {live_nonzero}"


def test_shaping_absent_when_scorer_none():
    """Default (no scorer) leaves metadata free of the shaping key."""
    mgr = TrajectoryManager()
    _two_turn_session(mgr, "sid", r1="a1", r2="a2")
    samples = mgr.get_trajectory("sid", base_sample=Sample(index=0, prompt=""), reward=1.0)
    assert len(samples) == 1
    assert "toolcall_turn_shaping" not in (samples[0].metadata or {})
    assert "toolcall_error_count" not in (samples[0].metadata or {})


def test_shaping_penalizes_errored_turn_only():
    """Scorer flags turn 2 (1 error); its total penalty -beta is spread over turn
    2's live tokens, turn 1 stays 0."""
    # scorer: 0 errors for first generated turn, 1 error for the second
    seen = []

    def scorer(node):
        seen.append(node)
        return 0 if len(seen) == 1 else 1

    mgr = TrajectoryManager(turn_scorer=scorer, shaping_beta=0.5, shaping_budget=100.0)
    _two_turn_session(mgr, "sid", r1="a1", r2="a2")
    samples = mgr.get_trajectory("sid", base_sample=Sample(index=0, prompt=""), reward=1.0)
    assert len(samples) == 1
    s = samples[0]
    vec = s.metadata["toolcall_turn_shaping"]
    assert len(vec) == s.response_length
    # response region = resp1(3) + prompt2 tail + resp2(2). Only resp2's 2 tokens
    # carry the turn's fixed -0.5 total, spread uniformly -> -0.25 each.
    assert vec[:3] == [0.0, 0.0, 0.0]  # turn 1 response, clean
    assert vec[-2:] == [-0.25, -0.25]  # turn 2 response: -beta/2 per live token
    assert abs(sum(vec[-2:]) + 0.5) < 1e-9  # turn total == -beta*errors
    # non-response prompt-tail tokens between the two responses are 0
    assert set(vec[3:-2]) <= {0.0}
    # the raw count sits next to the vector, in semantic units
    assert s.metadata["toolcall_error_count"] == 1


def test_budget_cap_scales_total():
    """Total |shaping| is capped at budget; per-turn fixed penalties scaled down
    proportionally."""

    def scorer(node):
        return 1  # every turn errs once

    mgr = TrajectoryManager(turn_scorer=scorer, shaping_beta=1.0, shaping_budget=1.0)
    _two_turn_session(mgr, "sid", r1="a1", r2="a2")
    samples = mgr.get_trajectory("sid", base_sample=Sample(index=0, prompt=""), reward=1.0)
    vec = samples[0].metadata["toolcall_turn_shaping"]
    total = sum(vec)
    # raw per-turn totals = -1.0 (turn1) + -1.0 (turn2) => total_abs 2.0; capped to
    # -1.0 (scale 0.5). Distributed: turn1 over 3 live tokens, turn2 over 2.
    assert abs(total + 1.0) < 1e-6
    # nonzero entries: 3 from turn1 (=-0.5/3) + 2 from turn2 (=-0.5/2)
    nonzero = [v for v in vec if v != 0.0]
    assert len(nonzero) == 5
    # each turn's scaled total is -0.5; proportions preserved across turns
    assert abs(sum(v for v in nonzero if abs(v - (-0.5 / 3)) < 1e-9) + 0.5) < 1e-9
    assert abs(sum(v for v in nonzero if abs(v - (-0.25)) < 1e-9) + 0.5) < 1e-9


def _single_turn_session(mgr, sid, *, response_ids, r="a1"):
    """One clean turn: system+user prompt, a trained response of the given ids."""
    p1 = SYS + USR
    mgr.record_turn(
        sid,
        turn=_turn(p1, response_ids),
        prompt_messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
        response_message=_asst_msg(r),
    )


def test_turn_length_invariance():
    """The whole point of the normalization: a single-error turn's TOTAL shaping is
    -beta regardless of how many trained tokens it emitted."""

    def scorer(node):
        return 1

    beta = 0.5
    # Short response (3 tokens).
    mgr_a = TrajectoryManager(turn_scorer=scorer, shaping_beta=beta, shaping_budget=100.0)
    _single_turn_session(mgr_a, "a", response_ids=[9001, 9002, 9003])
    vec_a = mgr_a.get_trajectory("a", base_sample=Sample(index=0, prompt=""))[0].metadata["toolcall_turn_shaping"]

    # Long response (10 tokens).
    mgr_b = TrajectoryManager(turn_scorer=scorer, shaping_beta=beta, shaping_budget=100.0)
    _single_turn_session(mgr_b, "b", response_ids=list(range(9001, 9011)))
    vec_b = mgr_b.get_trajectory("b", base_sample=Sample(index=1, prompt=""))[0].metadata["toolcall_turn_shaping"]

    assert sum(vec_a) == pytest.approx(-beta)
    assert sum(vec_b) == pytest.approx(-beta)
    # different lengths, identical totals
    assert len(vec_a) != len(vec_b)


def test_error_count_proportionality():
    """Same token length, 1 error vs 2 errors -> total scales 1:2."""

    def scorer_1(node):
        return 1

    def scorer_2(node):
        return 2

    beta = 0.5
    mgr_a = TrajectoryManager(turn_scorer=scorer_1, shaping_beta=beta, shaping_budget=100.0)
    _single_turn_session(mgr_a, "a", response_ids=[9001, 9002, 9003])
    vec_a = mgr_a.get_trajectory("a", base_sample=Sample(index=0, prompt=""))[0].metadata["toolcall_turn_shaping"]

    mgr_b = TrajectoryManager(turn_scorer=scorer_2, shaping_beta=beta, shaping_budget=100.0)
    _single_turn_session(mgr_b, "b", response_ids=[9001, 9002, 9003])
    vec_b = mgr_b.get_trajectory("b", base_sample=Sample(index=1, prompt=""))[0].metadata["toolcall_turn_shaping"]

    assert sum(vec_a) == pytest.approx(-beta)
    assert sum(vec_b) == pytest.approx(-2 * beta)
    assert sum(vec_b) == pytest.approx(2 * sum(vec_a))


def test_error_count_is_raw_and_budget_independent():
    """metadata["toolcall_error_count"] is the summed scorer output for the
    sample, reported before beta scaling and before the budget cap."""

    def scorer(node):
        return 3

    # Beta and budget both differ, but the semantic count must not: it is the
    # number of (other_error-excluded) errors the scorer found, full stop.
    mgr = TrajectoryManager(turn_scorer=scorer, shaping_beta=0.25, shaping_budget=0.01)
    _two_turn_session(mgr, "sid", r1="a1", r2="a2")
    s = mgr.get_trajectory("sid", base_sample=Sample(index=0, prompt=""), reward=1.0)[0]

    # Two scored turns x 3 errors each; the tiny budget only scales the vector.
    assert s.metadata["toolcall_error_count"] == 6
    assert sum(abs(v) for v in s.metadata["toolcall_turn_shaping"]) == pytest.approx(0.01)


def test_error_count_zero_for_clean_trajectory():
    """A scored-but-clean trajectory records an explicit 0, not a missing key."""

    def scorer(node):
        return 0

    mgr = TrajectoryManager(turn_scorer=scorer, shaping_beta=0.5, shaping_budget=100.0)
    _two_turn_session(mgr, "sid", r1="a1", r2="a2")
    s = mgr.get_trajectory("sid", base_sample=Sample(index=0, prompt=""), reward=1.0)[0]
    assert s.metadata["toolcall_error_count"] == 0


def test_adapter_forwards_scorer_to_manager():
    """BaseAdapter passes turn_scorer + scalars into its TrajectoryManager."""
    from slime.agent.adapters.common import BaseAdapter

    def scorer(node):
        return 0

    class _Tok:
        def apply_chat_template(self, *a, **k):
            return {"input_ids": [1]}

        def decode(self, *a, **k):
            return ""

    class _RoutelessAdapter(BaseAdapter):
        def _register_routes(self, app):
            pass

    ad = _RoutelessAdapter(
        tokenizer=_Tok(),
        sglang_url="http://x",
        turn_scorer=scorer,
        shaping_beta=0.25,
        shaping_budget=2.0,
    )
    assert ad.manager._turn_scorer is scorer
    assert ad.manager._shaping_beta == 0.25
    assert ad.manager._shaping_budget == 2.0


if __name__ == "__main__":
    pytest.main([__file__])
