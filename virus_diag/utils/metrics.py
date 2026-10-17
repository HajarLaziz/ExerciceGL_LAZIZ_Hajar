"""Métriques de classification binaire."""


def classification_metrics(expected, predicted):
    if not expected or len(expected) != len(predicted):
        raise ValueError("Les deux listes doivent être non vides et de même taille.")
    if any(value not in (0, 1) for value in [*expected, *predicted]):
        raise ValueError("Les étiquettes doivent valoir 0 ou 1.")
    tp = sum(y == 1 and p == 1 for y, p in zip(expected, predicted))
    fp = sum(y == 0 and p == 1 for y, p in zip(expected, predicted))
    fn = sum(y == 1 and p == 0 for y, p in zip(expected, predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "accuracy": sum(y == p for y, p in zip(expected, predicted)) / len(expected),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }
