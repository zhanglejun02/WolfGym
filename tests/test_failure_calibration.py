from dataclasses import replace

import pytest

from wolfgym.core.version import SIMULATOR_VERSION
from wolfgym.evaluation.failure_calibration import (
    LossEpisode, binomial_upper_bound, calibrate_loss_event, sustained_loss_score,
)


def episode(seed, loss=.1):
    return LossEpisode(f"trajectory-{seed}", seed, (0., loss, loss, 0.), True)


def calibrate(calibration, validation, **kwargs):
    return calibrate_loss_event(calibration, validation, source_version=SIMULATOR_VERSION,
                                context_id="fixed-test-market", **kwargs)


def test_sustained_score_ignores_single_day_spikes_and_requires_consecutive_days():
    assert sustained_loss_score([0, .5, 0, .5], 2) == 0
    assert sustained_loss_score([0, .5, .2, 0], 2) == .2
    assert sustained_loss_score([0, .5, 0, .5], 1) == .5


def test_insufficient_tail_sample_does_not_use_maximum_or_freeze_labels():
    report = calibrate([episode(1)], [episode(2)])
    assert report["definition"]["threshold"] is None
    assert not report["finite_threshold_available"]
    assert not report["release_eligible"]


def test_exact_heldout_upper_bound_not_zero_for_zero_observed_failures():
    report = calibrate([episode(i) for i in range(99)], [episode(100, 0)])
    assert report["definition"]["threshold"] == .1
    assert report["heldout_false_alarm"]["rate"] == 0
    assert report["heldout_false_alarm"]["one_sided_upper"] == pytest.approx(.95)
    assert not report["clean_validation_passed"]


def test_clean_statistical_pass_is_not_a_release_certificate():
    report = calibrate([episode(i) for i in range(99)], [episode(i, 0) for i in range(100, 500)])
    assert report["clean_validation_passed"]
    assert not report["release_eligible"]
    assert report["calibration_false_alarm"]["rate"] == 0  # strict > handles ties
    assert {(r["threshold_factor"], r["persistence"]) for r in report["sensitivity"]} == {
        (.8, 2), (1., 2), (1.2, 2), (1., 1)}


@pytest.mark.parametrize("validation", [[episode(0)], [replace(episode(1), seed=0)],
                                        [replace(episode(1), clean=False)]])
def test_leakage_and_attack_contamination_fail_closed(validation):
    with pytest.raises(ValueError):
        calibrate([episode(0)], validation)


def test_validation_never_changes_selected_threshold_or_criterion_identity():
    cal = [episode(i) for i in range(99)]
    first = calibrate(cal, [episode(100, 0)])
    second = calibrate(cal, [episode(100, 1)])
    assert first["criterion_identity"] == second["criterion_identity"]
    assert first["heldout_false_alarm"] != second["heldout_false_alarm"]
    assert first["criterion_identity"] != calibrate(cal, [episode(100)], persistence=1)["criterion_identity"]


def test_old_versions_and_incomplete_episodes_are_rejected():
    with pytest.raises(ValueError, match="version"):
        calibrate_loss_event([episode(0)], [episode(1)], source_version="old", context_id="x")
    with pytest.raises(ValueError, match="horizon"):
        calibrate([episode(0)], [replace(episode(1), losses=(0.,))])
    with pytest.raises(ValueError):
        sustained_loss_score([float("nan")], 1)
    with pytest.raises(ValueError):
        sustained_loss_score([.1], 2)


def test_exact_binomial_bound_matches_analytic_nonzero_cases():
    assert binomial_upper_bound(1, 2, .95) == pytest.approx(.95 ** .5)
    assert binomial_upper_bound(9, 10, .95) == pytest.approx(.95 ** .1)
    assert binomial_upper_bound(10, 10, .95) == 1
    assert binomial_upper_bound(0, 0, .95) == 1
    with pytest.raises(ValueError):
        binomial_upper_bound(.5, 2, .95)


