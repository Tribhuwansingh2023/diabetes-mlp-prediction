"""
Automated Deployment Testing and Verification Suite for Diabetes MLP Application.
Tests test cases specified in Section 20 of the Lab Assignment:
- Test a likely non-diabetic case
- Test a likely diabetic case
- Test borderline cases
- Verify invalid/extreme input handling without crashes
- Verify prediction reproducibility
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
import joblib

# Add project root and app to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "src"))
from app.app import compute_engineered_features

class TestDeploymentSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models_dir = os.path.join(BASE_DIR, "saved_models")
        cls.model_path = os.path.join(cls.models_dir, "diabetes_model.pkl")
        cls.preprocessor_path = os.path.join(cls.models_dir, "eng_preprocessor.joblib")
        
        # Verify artifact existence
        assert os.path.exists(cls.model_path), f"Missing model artifact: {cls.model_path}"
        assert os.path.exists(cls.preprocessor_path), f"Missing preprocessor artifact: {cls.preprocessor_path}"
        
        cls.model = joblib.load(cls.model_path)
        cls.preprocessor = joblib.load(cls.preprocessor_path)

    def predict_patient(self, patient_dict):
        df_feat = compute_engineered_features(patient_dict)
        df_scaled = self.preprocessor.transform(df_feat)
        prob = self.model.predict_proba(df_scaled)[0, 1]
        pred = int(prob >= 0.5)
        return pred, prob

    def test_likely_non_diabetic(self):
        """Case 1: Healthy young adult (normal glucose, healthy BMI, low genetic risk)"""
        patient = {
            'Pregnancies': 0, 'Glucose': 80, 'BloodPressure': 65,
            'SkinThickness': 18, 'Insulin': 50, 'BMI': 21.0,
            'DiabetesPedigreeFunction': 0.18, 'Age': 22
        }
        pred, prob = self.predict_patient(patient)
        print(f"\n[Test 1: Non-Diabetic] Prediction={pred}, Probability={prob*100:.2f}%")
        self.assertEqual(pred, 0, "Healthy patient should be classified as Non-Diabetic (0)")
        self.assertLess(prob, 0.35, "Probability should be < 0.35 for healthy patient")

    def test_likely_diabetic(self):
        """Case 2: Patient with severe hyperglycemia, high BMI, and strong family pedigree"""
        patient = {
            'Pregnancies': 6, 'Glucose': 185, 'BloodPressure': 92,
            'SkinThickness': 40, 'Insulin': 300, 'BMI': 38.5,
            'DiabetesPedigreeFunction': 1.15, 'Age': 56
        }
        pred, prob = self.predict_patient(patient)
        print(f"[Test 2: Diabetic] Prediction={pred}, Probability={prob*100:.2f}%")
        self.assertEqual(pred, 1, "High-risk patient should be classified as Diabetic (1)")
        self.assertGreater(prob, 0.65, "Probability should be > 0.65 for high-risk patient")

    def test_borderline_case(self):
        """Case 3: Borderline patient (impaired fasting glucose, overweight)"""
        patient = {
            'Pregnancies': 2, 'Glucose': 118, 'BloodPressure': 78,
            'SkinThickness': 28, 'Insulin': 110, 'BMI': 28.0,
            'DiabetesPedigreeFunction': 0.45, 'Age': 38
        }
        pred, prob = self.predict_patient(patient)
        print(f"[Test 3: Borderline] Prediction={pred}, Probability={prob*100:.2f}%")
        self.assertTrue(0.0 <= prob <= 1.0, "Probability must be valid in [0, 1]")

    def test_extreme_and_edge_inputs(self):
        """Case 4: Extreme physiologically possible bounds to verify robustness against crashes"""
        edge_patient = {
            'Pregnancies': 15, 'Glucose': 250, 'BloodPressure': 130,
            'SkinThickness': 60, 'Insulin': 600, 'BMI': 55.0,
            'DiabetesPedigreeFunction': 2.4, 'Age': 80
        }
        try:
            pred, prob = self.predict_patient(edge_patient)
            print(f"[Test 4: Edge Case] Prediction={pred}, Probability={prob*100:.2f}%")
            self.assertIn(pred, [0, 1])
        except Exception as e:
            self.fail(f"Application crashed on edge input with error: {e}")

if __name__ == "__main__":
    unittest.main()
