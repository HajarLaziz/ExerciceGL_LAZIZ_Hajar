"""Tests d'intégration et des utilitaires de la démonstration."""

from pathlib import Path
import unittest

from virus_diag.app.api import DiagnosisAPI
from virus_diag.core.dataset import Dataset
from virus_diag.core.logistic_regression import LogisticRegression
from virus_diag.core.neural_network import NeuralNetwork
from virus_diag.core.optimizer import GradientDescent
from virus_diag.pipeline.evaluator import Evaluator
from virus_diag.pipeline.trainer import Trainer
from virus_diag.utils.metrics import classification_metrics
from virus_diag.utils.preprocessing import preprocess


DATA = Path(__file__).resolve().parents[1] / "virus_diag" / "data" / "patient_data.csv"
HEALTHY = {"temperature": 36.6, "toux": 0, "fatigue": 0}
INFECTED = {"temperature": 39.1, "toux": 1, "fatigue": 1}


class PipelineTests(unittest.TestCase):
    def test_train_test_split_is_disjoint(self):
        train, test = Dataset.from_csv(DATA), Dataset.from_csv(DATA, "test")
        self.assertEqual((len(train.labels), len(test.labels)), (12, 4))
        self.assertTrue(all(patient not in train.patients for patient in test.patients))

    def test_both_models_work_through_same_pipeline(self):
        for model_class in (LogisticRegression, NeuralNetwork):
            with self.subTest(model=model_class.__name__):
                model = model_class()
                with self.assertRaises(RuntimeError):
                    model.predict(HEALTHY)
                Trainer(model).train(Dataset.from_csv(DATA))
                self.assertLess(model.predict(HEALTHY), 0.5)
                self.assertGreaterEqual(model.predict(INFECTED), 0.5)
                api = DiagnosisAPI(model)
                self.assertEqual(api.diagnose(HEALTHY), {"diagnostic": "Sain"})
                self.assertEqual(api.diagnose(INFECTED), {"diagnostic": "Infecté"})
                before = model.predict(INFECTED)
                scores = Evaluator().evaluate(model, Dataset.from_csv(DATA, "test"))
                self.assertEqual(before, model.predict(INFECTED))
                self.assertEqual(set(scores), {"accuracy", "precision", "recall", "f1"})
                self.assertTrue(all(0 <= score <= 1 for score in scores.values()))

    def test_preprocessing(self):
        self.assertEqual(preprocess({"temperature": 38.5, "toux": 1, "fatigue": 0}), [0.5, 1, 0])
        for field, value in (("temperature", float("nan")), ("temperature", 50), ("toux", 2), ("fatigue", -1)):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                preprocess({**HEALTHY, field: value})

    def test_metrics_known_confusion_matrix(self):
        self.assertEqual(classification_metrics([1, 1, 0, 0], [1, 0, 1, 0]),
                         {"accuracy": 0.5, "precision": 0.5, "recall": 0.5, "f1": 0.5})

    def test_metrics_no_positive_prediction(self):
        result = classification_metrics([0, 1], [0, 0])
        self.assertEqual(result["precision"], 0.0)
        self.assertEqual(result["f1"], 0.0)

    def test_invalid_training_arguments(self):
        trainer = Trainer(LogisticRegression())
        for epochs in (0, -1, 1.5, True):
            with self.subTest(epochs=epochs), self.assertRaises(ValueError):
                trainer.train(Dataset.from_csv(DATA), epochs)
        for dataset in (Dataset([], []), Dataset([HEALTHY], []), Dataset([HEALTHY], [2])):
            with self.subTest(dataset=dataset), self.assertRaises(ValueError):
                trainer.train(dataset)
        for rate in (0, -1, float("nan"), float("inf")):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                GradientDescent(rate)


if __name__ == "__main__":
    unittest.main()
