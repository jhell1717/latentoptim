import numpy as np
from tqdm import tqdm

from .shape import Circle, Triangle, Rectangle, Diamond, Heart, Oval, Star, Pentagon

class Generator:
    """_summary_
    """
    def __init__(self,resolution,num_shapes = 100):
        """_summary_

        Args:
            num_shapes (int, optional): _description_. Defaults to 100.
        """
        self.num_shapes = num_shapes
        self.shapes = []
        self.resolution = resolution

    def generate_shapes(self):
        """_summary_
        """
        for _ in tqdm(range(self.num_shapes)):
            shape_type = np.random.choice(
                ['Circle', 'Triangle', 'Rectangle', 'Diamond', 'Heart', 'Oval', 'Star', 'Pentagon'])

            if shape_type == 'Circle':
                shape = Circle(n_points=self.resolution)

            if shape_type == 'Triangle':
                shape = Triangle(n_points=self.resolution)

            if shape_type == 'Rectangle':
                shape = Rectangle(n_points=self.resolution)


            if shape_type == 'Diamond':
                shape = Diamond(n_points=self.resolution)


            if shape_type == 'Heart':
                shape = Heart(n_points=self.resolution)


            if shape_type == 'Star':
                shape = Star(n_points=self.resolution)


            if shape_type == 'Oval':
                shape = Oval(n_points=self.resolution)


            if shape_type == 'Pentagon':
                shape = Pentagon(n_points=self.resolution)

            self.shapes.append(shape)
        return self.shapes