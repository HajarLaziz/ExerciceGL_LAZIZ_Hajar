"""Solution de la question 1 : diagnostic avec un modèle déjà entraîné."""

import math
from numbers import Real


class ClinicalPredictor:
    """Accepte tout objet dont predict(patient_data) renvoie un score scalaire.

    Le modèle est fourni déjà entraîné. Aucune méthode d'entraînement n'est
    appelée ici et aucun héritage de Model n'est exigé (duck typing).
    """

    def __init__(self, model):
        if not callable(getattr(model, "predict", None)):
            raise TypeError("Le modèle doit fournir une méthode predict().")
        self._model = model

    def diagnose(self, patient_data):
        prediction = self._model.predict(patient_data)
        if isinstance(prediction, bool) or not isinstance(prediction, Real):
            raise TypeError("predict() doit renvoyer un score numérique scalaire.")
        if not math.isfinite(prediction) or not 0 <= prediction <= 1:
            raise ValueError("Le score doit être fini et compris entre 0 et 1.")
        return "Infecté" if prediction >= 0.5 else "Sain"
