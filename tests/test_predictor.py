"""Tests du contrat de l'exercice, sans dépendre d'un algorithme IA."""

import unittest

from virus_diag.pipeline.predictor import ClinicalPredictor


class FixedModel:
    """Doublure de test : ce n'est pas un modèle entraîné."""

    def __init__(self, score):
        self.score = score
        self.calls = []

    def predict(self, patient_data):
        self.calls.append(patient_data)
        return self.score

    def train(self, *args):
        raise AssertionError("Un diagnostic ne doit jamais entraîner le modèle.")


class ClinicalPredictorTests(unittest.TestCase):
    def test_scores_below_threshold_are_healthy(self):
        for score in (0.0, 0.2, 0.499999):
            with self.subTest(score=score):
                self.assertEqual(ClinicalPredictor(FixedModel(score)).diagnose({}), "Sain")

    def test_exact_threshold_is_infected(self):
        self.assertEqual(ClinicalPredictor(FixedModel(0.5)).diagnose({}), "Infecté")

    def test_scores_above_threshold_are_infected(self):
        for score in (0.500001, 0.9, 1.0):
            with self.subTest(score=score):
                self.assertEqual(ClinicalPredictor(FixedModel(score)).diagnose({}), "Infecté")

    def test_patient_is_forwarded_once_without_training(self):
        patient = {"temperature": 38.5}
        model = FixedModel(0.7)
        ClinicalPredictor(model).diagnose(patient)
        self.assertEqual(len(model.calls), 1)
        self.assertIs(model.calls[0], patient)

    def test_duck_typing_without_inheritance_or_train(self):
        class IndependentModel:
            def predict(self, data):
                return data["score"]

        self.assertEqual(ClinicalPredictor(IndependentModel()).diagnose({"score": 0.8}), "Infecté")

    def test_missing_predict_is_rejected(self):
        for model in (None, object(), type("Invalid", (), {"predict": 42})()):
            with self.subTest(model=model), self.assertRaises(TypeError):
                ClinicalPredictor(model)

    def test_non_scalar_scores_are_rejected(self):
        for score in (None, "0.8", [0.8], True):
            with self.subTest(score=score), self.assertRaises(TypeError):
                ClinicalPredictor(FixedModel(score)).diagnose({})

    def test_non_finite_or_out_of_range_scores_are_rejected(self):
        for score in (-0.01, 1.01, float("nan"), float("inf")):
            with self.subTest(score=score), self.assertRaises(ValueError):
                ClinicalPredictor(FixedModel(score)).diagnose({})


if __name__ == "__main__":
    unittest.main()
