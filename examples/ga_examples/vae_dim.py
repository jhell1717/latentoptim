import numpy as np
import matplotlib.pyplot as plt

import torch

import vae

from pyga import Individual, blended, gaussian


class ShapeVAE(Individual):

    _crossover_method = staticmethod(blended)
    _mutate_method = staticmethod(gaussian)

    target_shape = None
    model = None  # Class-level model storage

    def __init__(self, genes, model=None):
        super().__init__(genes)
        if model is not None:
            self.model = model
        elif ShapeVAE.model is None:
            raise ValueError("Model must be provided either at class level or instance level")
        else:
            self.model = ShapeVAE.model

    @classmethod
    def set_model(cls, model):
        cls.model = model

    @classmethod
    def set_target_shape(cls, points):
        if cls.target_shape is not None:
            raise AttributeError("Target shape already set")
        cls.target_shape = points

    def evaluate_fitness(self):
        """
        Lower distance -> higher fitness
        """
        points = self.model.decoder(torch.tensor(self.genes,dtype=torch.float32).unsqueeze(0)).detach().numpy().reshape(-1,2)
        compactness = vae.Metrics(points).compute_compactness()
        self._fitness = 1/(compactness)
        return self._fitness

    def plot(self, ax):
        points = self.model.decoder(torch.tensor(self.genes,dtype=torch.float32).unsqueeze(0)).detach().numpy().reshape(-1,2)
        # points = np.array(self.genes).reshape(-1, 2)
        x = np.append(points[:, 0], points[0, 0])
        y = np.append(points[:, 1], points[0, 1])
        ax.plot(x, y, "k-")
        ax.fill(x, y, "c", alpha=0.3)
        ax.set_aspect("equal")
        ax.axis("off")

    