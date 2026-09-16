"""
Automated Deployment Verification and Regression Test Suite.
Tests the authoritative inference pipeline against contract, stability,
and boundary requirements.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

# Add src to system path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from prediction import predict_patient, load_artifacts, validate_patient_input

class TestDeploymentSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.preprocessor, cls.config = load_artifacts()

    def test_artifacts_exist_and_load(self):
        """1. Verify that production artifacts exist, load cleanly, and have valid types."""
        self.assertIsNotNone(self.model, "Model artifact failed to load.")
        self.assertIsNotNone(self.preprocessor, "Preprocessor artifact failed to load.")
        self.assertIsInstance(self.config, dict, "Model configuration must be a valid dictionary.")

    def test_feature_dimensions(self):
        """2. Verify feature count compatibility between preprocessor and model."""
        raw_dim = self.preprocessor.n_features_in_
        proc_dim = self.preprocessor.n_features_out_
        model_dim = getattr(self.model, 'n_features_in_', None)
        
        self.assertEqual(raw_dim, 8, "Expected 8 raw input features.")
        self.assertEqual(proc_dim, 24, "Expected 24 processed features (8 raw + 16 engineered).")
        if model_dim is not None:
            self.assertEqual(proc_dim, model_dim, "Preprocessor output dim must equal model input dim.")

    def test_prediction_contract(self):
        """3. Verify prediction output structure, types, and probability bounds."""
        sample_patient = {
            'Pregnancies': 2, 'Glucose': 115, 'BloodPressure': 72,
            'SkinThickness': 24, 'Insulin': 90, 'BMI': 26.5,
            'DiabetesPedigreeFunction': 0.38, 'Age': 32
        }
        res = predict_patient(sample_patient, threshold=0.50)
        
        self.assertIn('predicted_class', res)
        self.assertIn('probability', res)
        self.assertIn('processed_feature_count', res)
        
        self.assertIn(res['predicted_class'], [0, 1], "Predicted class must be binary (0 or 1).")
        self.assertTrue(0.0 <= res['probability'] <= 1.0, "Probability must be in range [0, 1].")
        self.assertEqual(res['processed_feature_count'], 24, "Processed feature count must be 24.")
        self.assertEqual(res['predicted_class'], int(res['probability'] >= 0.50))

    def test_prediction_determinism(self):
        """4. Verify that identical inputs produce identical deterministic outputs."""
        sample = {
            'Pregnancies': 3, 'Glucose': 140, 'BloodPressure': 80,
            'SkinThickness': 30, 'Insulin': 150, 'BMI': 31.2,
            'DiabetesPedigreeFunction': 0.55, 'Age': 45
        }
        res1 = predict_patient(sample)
        res2 = predict_patient(sample)
        self.assertEqual(res1['probability'], res2['probability'], "Inference must be strictly deterministic.")
        self.assertEqual(res1['predicted_class'], res2['predicted_class'])

    def test_missing_and_zero_values_robustness(self):
        """5. Verify that unmeasured/zero biological values are cleanly imputed without crashing."""
        patient_with_zeros = {
            'Pregnancies': 0, 'Glucose': 0, 'BloodPressure': 0,
            'SkinThickness': 0, 'Insulin': 0, 'BMI': 0.0,
            'DiabetesPedigreeFunction': 0.25, 'Age': 25
        }
        try:
            res = predict_patient(patient_with_zeros)
            self.assertIn(res['predicted_class'], [0, 1])
            self.assertTrue(0.0 <= res['probability'] <= 1.0)
        except Exception as e:
            self.fail(f"Pipeline crashed on zero biological values: {e}")

    def test_extreme_physiological_bounds(self):
        """6. Verify that extreme but physiologically possible boundary values do not crash."""
        extreme_patient = {
            'Pregnancies': 16, 'Glucose': 320, 'BloodPressure': 135,
            'SkinThickness': 65, 'Insulin': 650, 'BMI': 58.0,
            'DiabetesPedigreeFunction': 2.3, 'Age': 82
        }
        try:
            res = predict_patient(extreme_patient)
            self.assertIn(res['predicted_class'], [0, 1])
        except Exception as e:
            self.fail(f"Pipeline crashed on extreme inputs: {e}")

    def test_invalid_input_rejection(self):
        """7. Verify that non-numeric inputs or missing fields raise a clear ValueError."""
        invalid_patient = {
            'Pregnancies': 'invalid_string', 'Glucose': 120, 'BloodPressure': 70,
            'SkinThickness': 20, 'Insulin': 80, 'BMI': 25.0,
            'DiabetesPedigreeFunction': 0.5, 'Age': 30
        }
        with self.assertRaises(ValueError):
            validate_patient_input(invalid_patient)

if __name__ == "__main__":
    unittest.main()
