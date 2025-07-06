import numpy as np

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

    @staticmethod
    def generate_closed_noisy_shape(num_points=100, radius=0.4, noise_scale=0.01):
        # Step 1: Generate a noisy ring
        base_points = num_points - 1
        angles = np.linspace(0, 2 * np.pi, base_points, endpoint=False)
        noise = np.random.normal(0, noise_scale, size=base_points)
        noisy_radius = radius + noise
        x = noisy_radius * np.cos(angles)
        y = noisy_radius * np.sin(angles)
        shape = np.stack((x, y), axis=1)

        # Step 2: Apply random rotation
        theta = np.random.uniform(0, 2 * np.pi)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                    [np.sin(theta),  np.cos(theta)]])
        shape = shape @ rotation_matrix.T

        # Step 3: Apply random skew (shear)
        skew_x = np.random.uniform(-0.75, 0.75)
        skew_y = np.random.uniform(-0.75, 0.75)
        shear_matrix = np.array([[1, skew_x],
                                [skew_y, 1]])
        shape = shape @ shear_matrix.T

        # Step 4: Normalize to [0, 1]
        min_vals = shape.min(axis=0)
        max_vals = shape.max(axis=0)
        shape = (shape - min_vals) / (max_vals - min_vals + 1e-8)

        # Step 5: Close the loop by appending the first point
        shape = np.vstack([shape, shape[0]])

        return shape  

    
