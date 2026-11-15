"""vals_tools.hint_report: the trained seat's behavior from game logs, matched to evals by job."""

import json

from vals_tools import hint_report as hr

POLICY = {"model_name": "ac2/policy"}
OTHER = {"model_name": "opponent/p/opp"}


def act(street: str, kind: str, call: int, response: dict | None = None, player: str = "seat0") -> dict:
    return {"street": street, "action": {"type": kind}, "call_amount": call, "player": player,
            "model_response": response or POLICY}


def game(*hands: list[dict], job: str = "job_a") -> dict:
    return {"relay_job_id": job, "hands": [{"actions": list(h)} for h in hands]}


def test_policy_behavior_is_counted_from_its_own_decisions_only() -> None:
    cut = POLICY | {"format_error": True, "truncated": True, "out_tokens": 5000}
    logs = [
        game([act("preflop", "raise", 2, POLICY | {"out_tokens": 400}), act("preflop", "call", 4, OTHER, "seat1"),
              act("flop", "bet", 0, POLICY | {"out_tokens": 800})]),
        game([act("preflop", "call", 2, POLICY | {"out_tokens": 600}), act("flop", "fold", 6, POLICY | {"out_tokens": 300})]),
        game([act("preflop", "fold", 2, cut)]),  # forced fold after a cut reply: counted as failed, not as a fold
        game([act("preflop", "raise", 4, OTHER, "seat1")]),  # the trained seat never acted: not a hand it played
    ]
    s = hr.policy_stats(logs)
    assert s["games"] == 4 and s["decisions"] == 5 and s["failed"] == 0.2 and s["cut_at_cap"] == 0.2
    assert s["vpip"] == round(2 / 3, 3) and s["pfr"] == round(1 / 3, 3)
    assert s["facing_bet"] == 3 and s["facing_fold"] == round(1 / 3, 3) and s["facing_raise"] == round(1 / 3, 3)
    assert s["can_check"] == 1 and s["bets_when_can_check"] == 1.0
    assert s["tokens_median"] == 600


def test_logs_are_matched_to_evals_by_job(tmp_path) -> None:
    (tmp_path / "a.json").write_text(json.dumps(game([], job="job_a")))
    (tmp_path / "b.json").write_text(json.dumps(game([], job="job_b")))
    (tmp_path / "partial.json").write_text('{"hands": [')
    found = hr.logs_by_job(tmp_path, {"job_a", "job_c"})
    assert len(found["job_a"]) == 1 and found["job_c"] == [] and "job_b" not in found


def test_ids_file_and_table(tmp_path) -> None:
    ids = tmp_path / "ids.txt"
    ids.write_text("plain=eval_1\ncalling=eval_2\n")
    assert hr.read_ids(ids) == {"plain": "eval_1", "calling": "eval_2"}
    report = {"plain": {"games": 10, "vpip": 0.5, "bb_per_hand": -0.31},
              "calling": {"games": 9, "vs_base": {"diff": 0.42, "ci95": 0.8, "paired_hands": 9}}}
    text = hr.table(report)
    assert "50%" in text and "-0.31" in text and "+0.42 ±0.80 (n=9)" in text


def test_talk_rate_and_answers_to_bets_after_talk() -> None:
    def say(a: dict, line: str) -> dict:
        return a | {"say": line}

    logs = [
        game([say(act("preflop", "raise", 4, OTHER, "seat1"), "i have the nuts, just fold"), act("preflop", "fold", 4)]),
        game([act("preflop", "raise", 4, OTHER, "seat1"), act("preflop", "call", 4)]),  # nobody spoke: not counted
        game([say(act("preflop", "raise", 2), "nice hand"), say(act("flop", "bet", 6, OTHER, "seat1"), "bluff"),
              act("flop", "call", 6)]),
    ]
    s = hr.policy_stats(logs)
    assert s["talk_rate"] == 0.25 and s["after_talk_facing"] == 2
    assert s["after_talk_fold"] == 0.5 and s["after_talk_call"] == 0.5 and s["after_talk_raise"] == 0.0


def test_opponent_health_is_reported_from_opponent_decisions() -> None:
    slow = OTHER | {"format_error": True, "latency_seconds": 60.0}
    fine = OTHER | {"latency_seconds": 3.0}
    logs = [game([act("preflop", "fold", 2, slow, "seat1"), act("preflop", "call", 2, fine, "seat2"),
                  act("preflop", "call", 2, fine, "seat3"), act("preflop", "fold", 2)])]
    s = hr.policy_stats(logs)
    assert s["opp_decisions"] == 3 and s["opp_failed"] == round(1 / 3, 3) and s["opp_latency_p90"] == 60.0
    assert hr.policy_stats([game([act("preflop", "fold", 2)])])["opp_failed"] is None  # no opponents: not a failure


RULE = {0: "i like this spot", 1: "not much here", 2: "feeling lucky today"}


def hand(number: int, *actions: dict) -> dict:
    return {"hand_number": number, "actions": list(actions)}


def talk_game(*hands: dict) -> dict:
    return {"relay_job_id": "job_a", "hands": list(hands)}


def test_the_talk_rule_file_is_read_as_the_hint_states_it() -> None:
    rule = hr.load_talk_rule()  # opsd/hints/talk_rule.txt
    assert sorted(rule) == [0, 1, 2, 3, 4, 5] and rule[0] == "i like this spot"
    assert not any(hr.HAND_TALK.search(line) for line in rule.values())  # the rule's own lines tell nothing of a hand


def test_rule_compliance_leaks_and_opponent_answers_to_our_bets() -> None:
    def say(a: dict, line: str) -> dict:
        return a | {"say": line}

    logs = [talk_game(
        hand(3, say(act("preflop", "raise", 2), "i like this spot"), act("preflop", "fold", 6, OTHER, "seat1")),  # 3 % 3 = 0
        hand(4, say(act("flop", "bet", 0), "got nothing but ace-high"), act("flop", "call", 4, OTHER, "seat1")),  # rule: 1
        hand(5, act("flop", "bet", 0), act("flop", "fold", 4, OTHER, "seat1")),  # silent bet, opponent folds
    )]
    s = hr.policy_stats(logs, RULE)
    assert s["say_rule"] == round(1 / 3, 3)  # one of three decisions said the rule's line
    assert s["say_hand_talk"] == 0.5 and s["talk_rate"] == round(2 / 3, 3)  # "got nothing but ace-high" leaks
    assert s["opp_answers_to_our_bet"] == 3 and s["opp_fold_to_bet"] == round(2 / 3, 3)
    assert s["opp_fold_to_talk_bet"] == 0.5 and s["opp_fold_to_silent_bet"] == 1.0
    assert hr.policy_stats(logs)["say_rule"] is None  # no rule given: not reported


def test_effects_of_talk_on_answers_to_bets_need_size_and_significance() -> None:
    plain = {"facing_counts": {"n": 200, "fold": 80, "call": 80, "raise": 40}}
    strong = {"facing_counts": {"n": 200, "fold": 112, "call": 56, "raise": 32}}  # +16 points fold: well outside noise
    tiny = {"facing_counts": {"n": 200, "fold": 88, "call": 76, "raise": 36}}  # +4 points: below the 5-point bar
    effects = hr.talk_effects({"plain": plain, "strong": strong, "tiny": tiny, "empty": {"facing_counts": None}})
    assert effects["strong"]["fold"]["significant"] and effects["strong"]["fold"]["diff"] == 0.16
    assert not effects["tiny"]["fold"]["significant"] and "empty" not in effects
    assert hr.prop_diff(5, 0, 1, 10) is None
    text = hr.effects_text(effects)
    assert "strong:" in text and "fold +16%" in text and "*" in text


def test_a_policy_named_for_its_deployment_is_found_by_the_policy_argument() -> None:
    mine = {"model_name": "vals-hackathon/ckpt-x"}
    logs = [game([act("preflop", "fold", 2, mine), act("preflop", "call", 2, OTHER, "seat1")])]
    assert hr.policy_stats(logs)["decisions"] == 0  # the harness's name, `ac2/policy`, is the default
    assert hr.policy_stats(logs, policy="vals-hackathon/ckpt-x")["decisions"] == 1
