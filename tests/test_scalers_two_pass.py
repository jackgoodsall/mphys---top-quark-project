import unittest

import numpy as np

from src.data_utils.scalers import LogMinMaxScaler


def fit_chunks(chunks):
    scaler = LogMinMaxScaler()
    for chunk in chunks:
        scaler.partial_fit_bounds(chunk)
    scaler.freeze_bounds()
    for chunk in chunks:
        scaler.partial_fit_standardization(chunk)
    return scaler


class TwoPassScalerTest(unittest.TestCase):
    def test_chunk_order_is_irrelevant(self):
        chunks = [np.array([[0.0], [2.0], [100.0]]), np.array([[1.0], [9.0], [30.0]])]
        forward = fit_chunks(chunks)
        reverse = fit_chunks(chunks[::-1])
        probe = np.array([[0.0], [3.0], [100.0]])
        np.testing.assert_allclose(forward.transform(probe), reverse.transform(probe), atol=1e-12)
        np.testing.assert_allclose(forward.inverse_transform(forward.transform(probe)), probe, atol=1e-10)

    def test_matches_one_shot_fit(self):
        chunks = [np.arange(5.0)[:, None], np.arange(5.0, 11.0)[:, None]]
        streamed = fit_chunks(chunks)
        one_shot = LogMinMaxScaler().fit(np.concatenate(chunks))
        probe = np.arange(11.0)[:, None]
        np.testing.assert_allclose(streamed.transform(probe), one_shot.transform(probe), atol=1e-12)

    def test_legacy_streaming_api_fails_loudly(self):
        scaler = LogMinMaxScaler()
        scaler.partial_fit([[1.0], [2.0]])
        with self.assertRaisesRegex(RuntimeError, "two-pass API"):
            scaler.partial_fit([[3.0]])


if __name__ == "__main__":
    unittest.main()
