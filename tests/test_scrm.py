import pytest

from metrics.SCRM import csv_eval, intersection_with_tolerance, union_with_tolerance


@pytest.mark.parametrize(
    ("prediction", "reference", "matches"),
    [
        pytest.param(10, -10, False, id="opposite-sign"),
        pytest.param(100, -10, False, id="large-positive-error"),
        pytest.param(-10, -10, True, id="exact-negative"),
        pytest.param(-10.4, -10, True, id="negative-within-five-percent"),
        pytest.param(-10.51, -10, False, id="negative-outside-five-percent"),
        pytest.param(10, 10, True, id="exact-positive"),
        pytest.param(10.51, 10, False, id="positive-outside-five-percent"),
        pytest.param(0, 0, True, id="exact-zero"),
        pytest.param(6e-8, 0, False, id="zero-reference-outside-epsilon-tolerance"),
        pytest.param(1.5e-7, 1e-7, True, id="near-zero-positive"),
        pytest.param(-4.5e-7, -5e-7, True, id="near-zero-negative"),
        pytest.param(-1e-6, -1e-6, True, id="negative-epsilon"),
    ],
)
def test_numeric_triple_overlap(prediction, reference, matches):
    predicted = [("region", "amount", prediction)]
    observed = [("region", "amount", reference)]
    assert intersection_with_tolerance(predicted, observed, 0, 0.05) == (
        predicted if matches else []
    )
    expected_union = set(observed if matches else predicted + observed)
    assert set(union_with_tolerance(predicted, observed, 0, 0.05)) == expected_union


@pytest.mark.parametrize(
    ("prediction", "tol_word", "matches"),
    [("north", 0, True), ("nort", 1, True), ("nort", 0, False)],
)
def test_text_triple_edit_distance(prediction, tol_word, matches):
    predicted = [("region", "direction", prediction)]
    reference = [("region", "direction", "north")]
    assert intersection_with_tolerance(predicted, reference, tol_word, 0.05) == (
        predicted if matches else []
    )
    expected_union = set(reference if matches else predicted + reference)
    assert set(union_with_tolerance(predicted, reference, tol_word, 0.05)) == expected_union


@pytest.mark.parametrize(("prediction", "expected"), [("100", 0.0), ("-10", 1.0)])
def test_negative_reference_csv_scores(prediction, expected):
    table = r"Region \t Amount \n North \t "
    scores, _ = csv_eval([table + prediction], [table + "-10"], easy=1)
    assert scores[:4] == pytest.approx((expected,) * 4)  # EM and three mAP levels


def test_repeated_matching_preserves_inputs():
    matched = ("region", "amount", -10.4)
    other_prediction = ("zone", "size", 1)
    predicted = [matched, other_prediction]
    reference = [("region", "amount", -10), ("zone", "size", 2)]
    original_predicted = predicted.copy()
    original_reference = reference.copy()

    for _ in range(2):
        assert intersection_with_tolerance(predicted, reference, 0, 0.05) == [matched]
        assert set(union_with_tolerance(predicted, reference, 0, 0.05)) == (
            {*original_reference, other_prediction}
        )

    assert predicted == original_predicted
    assert reference == original_reference
