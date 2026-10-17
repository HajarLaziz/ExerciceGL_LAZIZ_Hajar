"""Prétraitement déterministe partagé entre apprentissage et prédiction."""

import math


def preprocess(patient_data):
    """Convertit température, toux et fatigue en trois nombres entre 0 et 1."""
    temperature = float(patient_data["temperature"])
    cough = float(patient_data["toux"])
    fatigue = float(patient_data["fatigue"])
    if not math.isfinite(temperature) or not 35 <= temperature <= 42:
        raise ValueError("La température doit être comprise entre 35 et 42.")
    if cough not in (0, 1) or fatigue not in (0, 1):
        raise ValueError("Toux et fatigue doivent valoir 0 ou 1.")
    return [(temperature - 35) / 7, cough, fatigue]
