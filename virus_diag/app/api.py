"""Façade Python réutilisable ; ce module ne démarre pas de serveur HTTP."""

from virus_diag.pipeline.predictor import ClinicalPredictor


class DiagnosisAPI:
    def __init__(self, trained_model):
        self._predictor = ClinicalPredictor(trained_model)

    def diagnose(self, patient_data):
        return {"diagnostic": self._predictor.diagnose(patient_data)}
