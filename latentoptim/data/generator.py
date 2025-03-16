import numpy as np
from tqdm import tqdm

from .shape import Circle, Triangle, NoisyShape, Rectangle, Diamond, Heart, Oval, RotatedShape, Star, Pentagon

class Generator:
    """_summary_
    """
    def __init__(self,num_shapes = 100):
        """_summary_

        Args:
            num_shapes (int, optional): _description_. Defaults to 100.
        """
        self.num_shapes = num_shapes
        self.shapes = []

    def generate_shapes(self):
        """_summary_
        """
        for _ in tqdm(range(self.num_shapes)):
            shape_type = np.random.choice(
                ['Circle', 'Triangle', 'Rectangle', 'Diamond', 'Heart', 'Oval', 'Star', 'Pentagon'])

            if shape_type == 'Circle':
                shape = Circle(n_points=100)

            if shape_type == 'Triangle':
                shape = Triangle(n_points=100)

            if shape_type == 'Rectangle':
                shape = Rectangle(n_points=100)


            if shape_type == 'Diamond':
                shape = Diamond(n_points=100)


            if shape_type == 'Heart':
                shape = Heart(n_points=100)


            if shape_type == 'Star':
                shape = Star(n_points=100)


            if shape_type == 'Oval':
                shape = Oval(n_points=100)


            if shape_type == 'Pentagon':
                shape = Pentagon(n_points=100)

            if np.random.rand() > 0.5:
                shape = RotatedShape(shape, max_rotation=360)

            if np.random.rand() > 0.7:
                noise_fraction = np.random.uniform(0.1, 0.6)
                shape = NoisyShape(shape, noise_level=np.random.uniform(
                    0.01, 0.05), noise_fraction=noise_fraction)
            # shape.plot()
            # if np.random.rand() > 0.3:
            #     shape = Shape.skew(shape.points)

            self.shapes.append(shape)
        return self.shapes