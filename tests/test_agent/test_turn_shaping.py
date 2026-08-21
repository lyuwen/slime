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

    # (3) The budget denominator counts only live tokens. Turn 2 has 2 live,
    # errored tokens at beta=0.25 -> raw |shaping| = 0.5, which is under the 1.0
    # budget, so NO scale-down happens (budget is a cap, not a target). Crucially,
    # the 3 masked-out turn-1 tokens contribute 0 to the denominator: pre-fix they
    # would have added 0.75, forcing a spurious 1.0/1.25=0.8 scale-down and
    # smearing penalty onto masked tokens.
    total_abs = sum(abs(v) for v in vec)
    assert abs(total_abs - 0.5) < 1e-9, f"budget denominator included masked tokens: {total_abs}"
    live_nonzero = [v for v, m in zip(vec, s.loss_mask, strict=True) if m == 1 and v != 0.0]
    assert live_nonzero == [-0.25, -0.25], f"live tokens not penalized un-scaled: {live_nonzero}"


def test_shaping_absent_when_scorer_none():
    """Default (no scorer) leaves metadata free of the shaping key."""
    mgr = TrajectoryManager()
    _two_turn_session(mgr, "sid", r1="a1", r2="a2")
    samples = mgr.get_trajectory("sid", base_sample=Sample(index=0, prompt=""), reward=1.0)
    assert len(samples) == 1
    assert "toolcall_turn_shaping" not in (samples[0].metadata or {})


def test_shaping_penalizes_errored_turn_only():
    """Scorer flags turn 2 (1 error); its response tokens get -beta, turn 1 stays 0."""
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
    # response region = resp1(3) + prompt2 tail + resp2(2). Only resp2 tokens are -0.5.
    assert vec[:3] == [0.0, 0.0, 0.0]  # turn 1 response, clean
    assert vec[-2:] == [-0.5, -0.5]  # turn 2 response, 1 error * -0.5
    # non-response prompt-tail tokens between the two responses are 0
    assert set(vec[3:-2]) <= {0.0}


def test_budget_cap_scales_total():
    """Total |shaping| is capped at budget; proportions preserved."""

    def scorer(node):
        return 1  # every turn errs once

    mgr = TrajectoryManager(turn_scorer=scorer, shaping_beta=1.0, shaping_budget=1.0)
    _two_turn_session(mgr, "sid", r1="a1", r2="a2")
    samples = mgr.get_trajectory("sid", base_sample=Sample(index=0, prompt=""), reward=1.0)
    vec = samples[0].metadata["toolcall_turn_shaping"]
    total = sum(vec)
    # raw total = -(3 + 2) = -5 over 5 response tokens; capped to -1.0
    assert abs(total + 1.0) < 1e-6
    # all nonzero entries equal (uniform -beta before scaling), scaled uniformly
    nonzero = [v for v in vec if v != 0.0]
    assert len(nonzero) == 5
    assert all(abs(v - nonzero[0]) < 1e-9 for v in nonzero)


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
