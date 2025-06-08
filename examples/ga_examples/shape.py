import numpy as np
import matplotlib.pyplot as plt

import vae

from pyga import Individual, blended, gaussian


class ShapeOrig(Individual):

    _crossover_method = staticmethod(blended)
    _mutate_method = staticmethod(gaussian)

    target_shape = None

    def __init__(self, genes):
        super().__init__(genes)

    @classmethod
    def set_target_shape(cls, points):
        if cls.target_shape is not None:
            raise AttributeError("Target shape already set")
        cls.target_shape = points

    def evaluate_fitness(self):
        """
        Lower distance -> higher fitness
        """
        compactness = vae.Metrics(self.genes.reshape(-1,2)).compute_compactness()
        self._fitness = 1/(compactness)
        return self._fitness

    def plot(self, ax):
        points = np.array(self.genes).reshape(-1, 2)
        x = np.append(points[:, 0], points[0, 0])
        y = np.append(points[:, 1], points[0, 1])
        ax.plot(x, y, "k-")
        ax.fill(x, y, "c", alpha=0.3)
        ax.set_aspect("equal")
        ax.axis("off")

    
