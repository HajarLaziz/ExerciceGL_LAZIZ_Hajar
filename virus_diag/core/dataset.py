"""Chargement des données fictives et séparation explicite train/test."""

import csv
from dataclasses import dataclass

from virus_diag.utils.preprocessing import preprocess


@dataclass
class Dataset:
    patients: list
    labels: list

    @classmethod
    def from_csv(cls, path, split="train"):
        if split not in ("train", "test"):
            raise ValueError("Le groupe doit être train ou test.")
        patients, labels = [], []
        with open(path, encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                if row["split"] != split:
                    continue
                patient = {key: float(row[key]) for key in ("temperature", "toux", "fatigue")}
                preprocess(patient)
                label = int(row["infecte"])
                if label not in (0, 1):
                    raise ValueError("L'étiquette doit valoir 0 ou 1.")
                patients.append(patient)
                labels.append(label)
        if not patients:
            raise ValueError("Le jeu de données est vide.")
        return cls(patients, labels)
