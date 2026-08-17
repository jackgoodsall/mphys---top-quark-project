import numpy as np
import unittest

from src.analysis.evaluate_chain import decode_legal


def test_legal_decoder_enforces_subset_and_cross_chain_disjointness():
    top = np.array([[[4, 3, 2, -2], [4, 3, 2, -2]]], dtype=float)
    W = np.array([[[9, -2, 1, 8], [-2, 9, 1, 8]]], dtype=float)
    valid = np.array([[True, True, True, False]])

    pred_top, pred_W = decode_legal(top, W, valid, fixed_cardinality=False)

    assert not np.any(pred_top[:, 0] & pred_top[:, 1])
    assert not np.any(pred_W[:, 0] & pred_W[:, 1])
    assert np.all(~pred_W | pred_top)
    assert not pred_top[:, :, 3].any()
    assert not pred_W[:, :, 3].any()


def test_legal_decoder_supports_default_fixed_cardinality():
    top = np.full((1, 2, 6), -5.0)
    W = np.full((1, 2, 6), -5.0)
    top[0, 0, :3] = 5
    top[0, 1, 3:] = 5
    W[0, 0, :2] = 5
    W[0, 1, 3:5] = 5

    pred_top, pred_W = decode_legal(top, W, np.ones((1, 6), dtype=bool))

    assert pred_top.sum(axis=-1).tolist() == [[3, 3]]
    assert pred_W.sum(axis=-1).tolist() == [[2, 2]]
    assert np.all(~pred_W | pred_top)


def test_empty_and_padded_events_never_select_invalid_jets():
    top = np.array([
        [[10, 10, 10, 100], [10, 10, 10, 100]],
        [[10, 100, 100, 100], [10, 100, 100, 100]],
    ], dtype=float)
    W = top.copy()
    valid = np.array([[True, True, True, False], [True, False, False, False]])

    pred_top, pred_W = decode_legal(top, W, valid)

    assert not pred_top[0, :, 3].any()
    assert not pred_top[1, :, 1:].any()
    assert not pred_W[0, :, 3].any()
    assert not pred_W[1, :, 1:].any()


def test_threshold_candidate_wins_when_fixed_cardinality_would_add_bad_jets():
    top = np.full((1, 2, 6), -2.0)
    W = np.full((1, 2, 6), -2.0)
    top[0, 0, 0] = 2
    top[0, 1, 1] = 2
    W[0, 0, 0] = 2
    W[0, 1, 1] = 2

    pred_top, pred_W = decode_legal(top, W, np.ones((1, 6), dtype=bool))

    assert pred_top.sum() == 2
    assert pred_W.sum() == 2
    assert pred_top[0, 0, 0] and pred_top[0, 1, 1]
    assert np.all(~pred_W | pred_top)


def load_tests(loader, tests, pattern):
    """Expose the historical pytest-style functions to stdlib discovery."""
    suite = unittest.TestSuite()
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            suite.addTest(unittest.FunctionTestCase(value, description=name))
    return suite
