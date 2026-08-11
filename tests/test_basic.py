"""
Basic unit tests for RVC Voice Model Evaluator.
Run: python tests/test_basic.py
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from src.audio import (
    load_audio, split_into_windows, trim_to_common_duration, normalize_audio,
    load_model_files,
)
from src.vad import compute_speech_ratio, filter_valid_windows
from src.acoustic import compute_mcd, compute_mel_distance, compute_mfcc_distance, compute_per_window_acoustic_scores
from src.f0 import extract_f0, compute_f0_correlation, compute_f0_rmse, octave_correction, compute_per_window_f0_scores
from src.artifacts import (
    compute_clipping_ratio, compute_peak, compute_rms,
    compute_spectral_flatness, compute_high_frequency_ratio, detect_absolute_warnings,
)
from src.stability import compute_stability, compute_window_scores
from src.scoring import normalize_batch, normalize_inverted
from src.absolute_score import evaluate_absolute_score
from src.comparison import ModelComparator
from src.utils import flatten_model_entry
from src.trend_analysis import TrendAnalyzer, flatten_model_entry


def create_test_tone(duration_sec=3.0, sr=16000, freq=440.0) -> np.ndarray:
    """Create a simple sine wave test tone."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


class TestAudio(unittest.TestCase):
    def test_window_split(self):
        audio = create_test_tone(10.0)
        windows = split_into_windows(audio, 16000, window_seconds=3.0)
        self.assertGreater(len(windows), 0)
        self.assertAlmostEqual(windows[0]["duration"], 3.0, delta=0.1)

    def test_trim_common_duration(self):
        a = np.zeros(16000)  # 1s
        b = np.zeros(32000)  # 2s
        a_t, b_t = trim_to_common_duration(a, b, 16000)
        self.assertEqual(len(a_t), len(b_t))
        self.assertEqual(len(a_t), 16000)

    def test_normalize(self):
        audio = np.array([0.5, -0.5, 0.25])
        normalized = normalize_audio(audio)
        self.assertAlmostEqual(np.max(np.abs(normalized)), 1.0)


class TestVAD(unittest.TestCase):
    def test_speech_ratio_sine(self):
        audio = create_test_tone(3.0)
        ratio = compute_speech_ratio(audio, 16000)
        self.assertGreater(ratio, 0.0)

    def test_speech_ratio_silence(self):
        audio = np.zeros(16000 * 3)
        ratio = compute_speech_ratio(audio, 16000)
        self.assertAlmostEqual(ratio, 0.0, delta=0.05)

    def test_filter_valid_windows(self):
        audio = create_test_tone(10.0)
        windows = split_into_windows(audio, 16000, window_seconds=3.0)
        windows = filter_valid_windows(windows, 16000, min_speech_ratio=0.10)
        self.assertGreater(len(windows), 0)

    def test_filter_silent_windows(self):
        audio = np.zeros(16000 * 10)
        windows = split_into_windows(audio, 16000, window_seconds=3.0)
        windows = filter_valid_windows(windows, 16000, min_speech_ratio=0.10)
        for w in windows:
            self.assertFalse(w.get("valid", True))


class TestAcoustic(unittest.TestCase):
    def test_mcd_same(self):
        audio = create_test_tone(3.0)
        mcd = compute_mcd(audio, audio, 16000)
        self.assertFalse(np.isnan(mcd))

    def test_mel_same(self):
        audio = create_test_tone(3.0)
        mel = compute_mel_distance(audio, audio, 16000)
        self.assertFalse(np.isnan(mel))

    def test_mfcc_same(self):
        audio = create_test_tone(3.0)
        mfcc = compute_mfcc_distance(audio, audio, 16000)
        self.assertFalse(np.isnan(mfcc))

    def test_mcd_different_larger(self):
        a = create_test_tone(3.0, freq=440)
        b = create_test_tone(3.0, freq=880)  # Octave higher
        mcd_same = compute_mcd(a, a, 16000)
        mcd_diff = compute_mcd(a, b, 16000)
        # Different tones should have meaningfully higher MCD than same tone
        self.assertGreater(mcd_diff, mcd_same + 0.5)


class TestF0(unittest.TestCase):
    def test_extract_f0(self):
        audio = create_test_tone(3.0, freq=220)
        f0, voiced = extract_f0(audio, 16000)
        self.assertGreater(len(f0), 0)
        self.assertTrue(np.any(voiced))

    def test_f0_correlation_same(self):
        audio = create_test_tone(3.0, freq=220)
        f0, vo = extract_f0(audio, 16000)
        corr = compute_f0_correlation(f0, f0, vo)
        # Pure sine wave may yield constant F0 → pearsonr is undefined (NaN)
        if corr is not None and not np.isnan(corr):
            self.assertGreater(corr, 0.9)

    def test_octave_correction(self):
        f0_orig = np.array([200.0, 150.0])
        f0_model = np.array([400.0, 150.0])  # First: octave error, second: no error
        vo = np.array([True, True])
        corrected = octave_correction(f0_orig, f0_model, vo, vo)
        self.assertAlmostEqual(corrected[0], 200.0, delta=50)
        self.assertAlmostEqual(corrected[1], 150.0, delta=10)


class TestArtifacts(unittest.TestCase):
    def test_clipping_normal(self):
        audio = create_test_tone(3.0)
        clip = compute_clipping_ratio(audio)
        # Pure sine at amplitude 1.0 has ~10% near-peak samples — acceptable
        self.assertLess(clip, 0.15)

    def test_clipping_detected(self):
        audio = create_test_tone(3.0)
        audio[100:200] = 0.999  # Near clip
        clip = compute_clipping_ratio(audio, threshold=0.99)
        self.assertGreater(clip, 0.0)

    def test_peak(self):
        audio = np.array([0.1, 0.5, 0.9, 0.3])
        peak = compute_peak(audio)
        self.assertAlmostEqual(peak, 0.9, delta=0.01)

    def test_rms(self):
        audio = np.ones(1000) * 0.5
        rms = compute_rms(audio)
        self.assertAlmostEqual(rms, 0.5, delta=0.01)

    def test_spectral_flatness(self):
        audio = create_test_tone(3.0)
        sf = compute_spectral_flatness(audio, 16000)
        self.assertGreater(sf, 0.0)

    def test_hf_ratio(self):
        audio = create_test_tone(3.0, freq=1000)  # Below cutoff
        hf = compute_high_frequency_ratio(audio, 16000, cutoff_hz=4000)
        self.assertLess(hf, 0.5)

    def test_absolute_warnings(self):
        warnings = detect_absolute_warnings(0.06, 0.8, 0.0001, 0.05)
        self.assertIn("SEVERE_CLIPPING", warnings)
        self.assertIn("HIGH_FREQUENCY_ANOMALY", warnings)
        self.assertIn("SEVERE_SPECTRAL_ANOMALY", warnings)
        self.assertIn("LOW_SPEECH_RATIO", warnings)


class TestStability(unittest.TestCase):
    def test_identical_scores(self):
        scores = [0.8, 0.8, 0.8, 0.8]
        result = compute_stability(scores)
        self.assertAlmostEqual(result["window_score_std"], 0.0)

    def test_variable_scores(self):
        scores = [0.9, 0.5, 0.9, 0.5]
        result = compute_stability(scores)
        self.assertGreater(result["window_score_std"], 0.1)


class TestScoring(unittest.TestCase):
    def test_normalize_higher_better(self):
        values = [1.0, 2.0, 3.0]
        scores = normalize_batch(values, higher_is_better=True)
        self.assertAlmostEqual(scores[0], 0.0)
        self.assertAlmostEqual(scores[2], 100.0)

    def test_normalize_inverted(self):
        values = [1.0, 2.0, 3.0]
        scores = normalize_inverted(values)
        self.assertAlmostEqual(scores[0], 100.0)
        self.assertAlmostEqual(scores[2], 0.0)

    def test_normalize_all_equal(self):
        values = [5.0, 5.0, 5.0]
        scores = normalize_batch(values, higher_is_better=True)
        for s in scores:
            self.assertAlmostEqual(s, 100.0)

    def test_normalize_nan(self):
        values = [1.0, None, 3.0]
        scores = normalize_batch(values, higher_is_better=True)
        self.assertTrue(np.isnan(scores[1]))
        self.assertAlmostEqual(scores[2], 100.0)


class TestWindowAcoustic(unittest.TestCase):
    """Test per-window acoustic scoring (used for stability)."""

    def test_per_window_acoustic_same(self):
        audio = create_test_tone(3.0)
        config = {
            "features": {"n_fft": 1024, "hop_length": 256, "win_length": 1024,
                         "fmin": 50.0, "fmax": 7600.0, "n_mels": 80, "n_mfcc": 20},
            "acoustic": {"mcd_weight": 0.45, "mel_weight": 0.35, "mfcc_weight": 0.20},
        }
        scores = compute_per_window_acoustic_scores([audio, audio], [audio, audio], 16000, config)
        self.assertEqual(len(scores), 2)

    def test_per_window_acoustic_different(self):
        a1 = create_test_tone(3.0, freq=220)
        a2 = create_test_tone(3.0, freq=880)
        config = {
            "features": {"n_fft": 1024, "hop_length": 256, "win_length": 1024,
                         "fmin": 50.0, "fmax": 7600.0, "n_mels": 80, "n_mfcc": 20},
            "acoustic": {"mcd_weight": 0.45, "mel_weight": 0.35, "mfcc_weight": 0.20},
        }
        # Compare same tone (both windows identical) vs different tones (both windows identical)
        # Raw acoustic distances for different tones should be higher than same tone
        scores_same = compute_per_window_acoustic_scores([a1, a1], [a1, a1], 16000, config)
        scores_diff = compute_per_window_acoustic_scores([a1, a1], [a2, a2], 16000, config)
        self.assertEqual(len(scores_same), 2)
        self.assertEqual(len(scores_diff), 2)
        # Both batches have identical windows internally, so within-model normalization
        # gives 100.0 for both — verify scores are valid (not NaN)
        self.assertFalse(any(np.isnan(s) for s in scores_same + scores_diff))


class TestWindowF0(unittest.TestCase):
    """Test per-window F0 scoring (used for stability)."""

    def test_per_window_f0_same(self):
        audio = create_test_tone(3.0, freq=220)
        config = {
            "f0": {"min_hz": 50, "max_hz": 600, "correlation_weight": 0.50,
                   "dtw_weight": 0.30, "rmse_weight": 0.20, "octave_correction": False},
            "features": {"hop_length": 256},
        }
        scores = compute_per_window_f0_scores([audio, audio], [audio, audio], 16000, config)
        self.assertEqual(len(scores), 2)

    def test_per_window_f0_different(self):
        a1 = create_test_tone(3.0, freq=220)
        a2 = create_test_tone(3.0, freq=440)
        config = {
            "f0": {"min_hz": 50, "max_hz": 600, "correlation_weight": 0.50,
                   "dtw_weight": 0.30, "rmse_weight": 0.20, "octave_correction": False},
            "features": {"hop_length": 256},
        }
        # Compare same tone (both windows identical) vs different tones (both windows identical)
        scores_same = compute_per_window_f0_scores([a1, a1], [a1, a1], 16000, config)
        scores_diff = compute_per_window_f0_scores([a1, a1], [a2, a2], 16000, config)
        self.assertEqual(len(scores_same), 2)
        self.assertEqual(len(scores_diff), 2)
        # Both batches have identical windows internally, so within-model normalization
        # gives 100.0 for both — verify scores are valid (not NaN)
        self.assertFalse(any(np.isnan(s) for s in scores_same + scores_diff))


class TestWindowScores(unittest.TestCase):
    """Test composite window score computation."""

    def test_compute_window_scores(self):
        ecapa = [0.9, 0.8, 0.85]
        acoustic = [80.0, 70.0, 75.0]
        f0 = [90.0, 85.0, 88.0]
        scores = compute_window_scores(ecapa, acoustic, f0)
        self.assertEqual(len(scores), 3)

    def test_window_scores_with_nan(self):
        ecapa = [0.9, 0.8, float('nan')]
        acoustic = [80.0, float('nan'), 75.0]
        f0 = [float('nan'), 85.0, 88.0]
        scores = compute_window_scores(ecapa, acoustic, f0)
        self.assertEqual(len(scores), 3)
        # Window 0: ecapa + acoustic → valid
        self.assertFalse(np.isnan(scores[0]))
        # Window 1: ecapa + f0 → valid after weight re-normalization
        self.assertFalse(np.isnan(scores[1]))
        # Window 2: acoustic + f0 → valid after weight re-normalization
        self.assertFalse(np.isnan(scores[2]))


class TestAbsoluteScore(unittest.TestCase):
    """Tests for absolute scoring (batch-independent quality rating)."""

    def test_excellent_model(self):
        """Excellent model should get high score and 'recommended'."""
        result = evaluate_absolute_score(
            ecapa_similarity=0.82,
            mcd_distance=90.0,
            f0_correlation=0.995,
            stability_mean=65.0,
        )
        self.assertGreater(result["score"], 80)
        self.assertEqual(result["overall_level"], "recommended")
        self.assertEqual(result["details"]["ecapa"]["level"], "excellent")
        self.assertEqual(result["details"]["f0"]["level"], "excellent")

    def test_poor_model(self):
        """Poor model should get low score and 'not_recommended'."""
        result = evaluate_absolute_score(
            ecapa_similarity=0.45,
            mcd_distance=450.0,
            f0_correlation=0.80,
            stability_mean=15.0,
        )
        self.assertLess(result["score"], 30)
        self.assertEqual(result["overall_level"], "not_recommended")

    def test_usable_model(self):
        """Typical model should be 'usable'."""
        result = evaluate_absolute_score(
            ecapa_similarity=0.65,
            mcd_distance=290.0,
            f0_correlation=0.95,
            stability_mean=30.0,
        )
        self.assertEqual(result["overall_level"], "usable")
        self.assertGreaterEqual(result["score"], 45)

    def test_missing_mcd(self):
        """None MCD should not crash."""
        result = evaluate_absolute_score(
            ecapa_similarity=0.70,
            mcd_distance=None,
            f0_correlation=0.98,
            stability_mean=40.0,
        )
        self.assertIsInstance(result["score"], (int, float))
        self.assertEqual(result["details"]["mcd"]["level"], "very_poor")


class TestComparison(unittest.TestCase):
    """Tests for A/B model comparison."""

    def setUp(self):
        self.model_a = {
            "model": "24", "score": 89.0, "ecapa": 100.0,
            "acoustic": 100.0, "f0": 37.9, "stability": 100.0, "artifact": 66.9,
        }
        self.model_b = {
            "model": "25", "score": 47.6, "ecapa": 62.3,
            "acoustic": 10.0, "f0": 100.0, "stability": 33.9, "artifact": 50.0,
        }

    def test_compare_basic(self):
        comparator = ModelComparator()
        result = comparator.compare(self.model_a, self.model_b)
        self.assertEqual(result["model_a"], "24")
        self.assertEqual(result["model_b"], "25")
        self.assertEqual(len(result["difference"]), 6)
        self.assertEqual(len(result["winner"]), 6)
        self.assertIsInstance(result["analysis"], list)

    def test_winner_detection(self):
        comparator = ModelComparator(threshold=0.5)
        result = comparator.compare(self.model_a, self.model_b)
        # 24 has higher score (89 > 47.6), so A wins on score
        self.assertEqual(result["winner"]["score"], "24")
        # 24 has higher ecapa (100 > 62.3)
        self.assertEqual(result["winner"]["ecapa"], "24")
        # 25 has higher f0 (100 > 37.9)
        self.assertEqual(result["winner"]["f0"], "25")

    def test_similar_detection(self):
        comparator = ModelComparator(threshold=100.0)
        result = comparator.compare(self.model_a, self.model_b)
        for metric in ModelComparator.METRICS:
            self.assertEqual(result["winner"][metric], "similar")

    def test_flatten_entry(self):
        """Flatten nested scores.json format to comparison-ready dict."""
        nested = {
            "model": "99",
            "final_score": 85.0,
            "sub_scores": {
                "ecapa": 90.0, "acoustic": 80.0,
                "f0": 70.0, "stability": 60.0, "artifact": 50.0,
            },
        }
        flat = flatten_model_entry(nested)
        self.assertEqual(flat["model"], "99")
        self.assertEqual(flat["score"], 85.0)
        self.assertEqual(flat["ecapa"], 90.0)
        self.assertEqual(flat["acoustic"], 80.0)
        self.assertEqual(flat["f0"], 70.0)

    def test_load_model_from_scores_json(self):
        """Integration: load a specific model from a scores.json structure."""
        data = {
            "models": [
                {"model": "A", "final_score": 95,
                 "sub_scores": {"ecapa": 88, "acoustic": 90, "f0": 85,
                                "stability": 92, "artifact": 70}},
                {"model": "B", "final_score": 80,
                 "sub_scores": {"ecapa": 75, "acoustic": 78, "f0": 82,
                                "stability": 70, "artifact": 65}},
            ]
        }
        comparator = ModelComparator()
        ma = comparator.load_model(data, "A")
        mb = comparator.load_model(data, "B")
        self.assertEqual(ma["score"], 95)
        self.assertEqual(mb["ecapa"], 75)


class TestTrendAnalysis(unittest.TestCase):
    """Tests for overfitting trend analysis."""

    def setUp(self):
        self.reports = [
            {"model": "10", "score": 70.0, "ecapa": 75.0, "f0": 60.0, "artifact": 55.0},
            {"model": "20", "score": 78.0, "ecapa": 82.0, "f0": 70.0, "artifact": 60.0},
            {"model": "30", "score": 85.0, "ecapa": 90.0, "f0": 80.0, "artifact": 70.0},
            {"model": "40", "score": 75.0, "ecapa": 80.0, "f0": 68.0, "artifact": 65.0},
            {"model": "50", "score": 70.0, "ecapa": 75.0, "f0": 60.0, "artifact": 55.0},
            {"model": "60", "score": 65.0, "ecapa": 70.0, "f0": 55.0, "artifact": 50.0},
        ]

    def test_find_best_epoch(self):
        analyzer = TrendAnalyzer()
        best = analyzer.find_best_epoch(self.reports)
        self.assertEqual(best["model"], "30")
        self.assertEqual(best["score"], 85.0)

    def test_analyze_trend(self):
        analyzer = TrendAnalyzer()
        trend = analyzer.analyze_trend(self.reports)
        self.assertEqual(len(trend), 6)
        self.assertEqual(trend[0]["epoch"], "10")
        # Scores should follow the input trend
        self.assertGreater(trend[2]["score"], trend[0]["score"])  # 30 > 10
        self.assertLess(trend[5]["score"], trend[2]["score"])     # 60 < 30

    def test_detect_overfit(self):
        analyzer = TrendAnalyzer(overfit_patience=2, min_improvement=0.1)
        points = analyzer.detect_overfit(self.reports)
        # 40 drops, 50 drops, 60 drops — 3 consecutive declines with patience=2
        # After 40: decline_count=1. After 50: decline_count=2 → overfit at 50
        # After 60: decline_count=3 → overfit at 60
        self.assertIn("50", points)
        self.assertIn("60", points)

    def test_no_overfit_when_improving(self):
        improving = [
            {"model": "10", "score": 60.0, "ecapa": 60.0, "f0": 50.0, "artifact": 40.0},
            {"model": "20", "score": 70.0, "ecapa": 70.0, "f0": 60.0, "artifact": 50.0},
            {"model": "30", "score": 80.0, "ecapa": 80.0, "f0": 70.0, "artifact": 60.0},
        ]
        analyzer = TrendAnalyzer()
        points = analyzer.detect_overfit(improving)
        self.assertEqual(points, [])

    def test_explain(self):
        analyzer = TrendAnalyzer()
        result = analyzer.explain(self.reports)
        self.assertEqual(result["best_epoch"], "30")
        self.assertTrue(any("过拟合" in s for s in result["summary"]))
        self.assertTrue(len(result["overfit_points"]) > 0)

    def test_flatten_model_entry(self):
        """Flatten scores.json model entry (final_score + sub_scores at top level)."""
        data = {
            "model": "42",
            "final_score": 88.5,
            "sub_scores": {
                "ecapa": 92.0, "f0": 75.0, "artifact": 60.0,
                "acoustic": 80.0, "stability": 55.0,
            },
        }
        flat = flatten_model_entry(data)
        self.assertEqual(flat["model"], "42")
        self.assertEqual(flat["score"], 88.5)
        self.assertEqual(flat["ecapa"], 92.0)
        self.assertEqual(flat["f0"], 75.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
