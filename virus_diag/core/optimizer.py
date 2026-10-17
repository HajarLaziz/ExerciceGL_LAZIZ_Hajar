"""Descente de gradient utilisée par les deux modèles."""

import math


class GradientDescent:
    def __init__(self, learning_rate=0.1):
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("Le taux d'apprentissage doit être positif et fini.")
        self.learning_rate = learning_rate

    def update(self, value, gradient):
        return value - self.learning_rate * gradient
