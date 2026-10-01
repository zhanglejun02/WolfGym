"""Combination-task event: a frozen union, with each harm component reported."""
from wolfgym.evaluation.composite_harm import score_composite_harm
from wolfgym.evaluation.task_validation import damage_series, damage_event


def score_combination(reference, actual, reference_rows, actual_rows, threshold):
    damage = damage_series(reference, actual)
    wealth = damage_event(damage["H"], damage["G"], threshold)
    legacy = score_composite_harm(reference, actual, reference_rows, actual_rows)
    first = {"wealth": wealth["first_day"], "tail": legacy["first_days"]["tail"],
             "function": legacy["first_days"]["function"]}
    union = min((day for day in first.values() if day is not None), default=None)
    return {"damage": damage, "event": union is not None, "first_event_day": union,
            "components": {key: day is not None for key, day in first.items()}, "first_days": first,
            "definition": "sustained_wealth_OR_severe_tail_OR_sustained_function_v5",
            "horizon_complete": len(actual) == 30}
