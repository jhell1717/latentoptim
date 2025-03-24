import numpy as np
from tqdm import tqdm

from .shape import Circle, Triangle, Rectangle, Diamond, Heart, Oval, Star, Pentagon, Square


class Generator:
    """_summary_
    """

    def __init__(self, resolution, num_shapes=100):
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
                ['Circle', 'Triangle', 'Rectangle', 'Diamond', 'Heart', 'Oval', 'Star', 'Pentagon', 'Square'])

            if shape_type == 'Circle':
                shape = Circle(radius=1, n_points=self.resolution)

            if shape_type == 'Triangle':
                shape = Triangle(vertices=np.array([[0,0],[1,0],[0.5,np.sqrt(3)/2]]),n_points=self.resolution)

            if shape_type == 'Rectangle':
                shape = Rectangle(width=3, height=1, n_points=self.resolution)

            if shape_type == 'Diamond':
                shape = Diamond(width=1, height=2, n_points=self.resolution)

            if shape_type == 'Square':
                shape = Square(side_length=1, n_points=self.resolution)

            if shape_type == 'Heart':
                shape = Heart(n_points=self.resolution)

            if shape_type == 'Star':
                shape = Star(n_arms=5, outer_radius=1,
                             inner_radius=0.5, n_points=self.resolution)

            if shape_type == 'Oval':
                shape = Oval(major_axis=1,minor_axis=3,n_points=self.resolution)

            if shape_type == 'Pentagon':
                shape = Pentagon(radius=1,n_points=self.resolution)

            self.shapes.append(shape)
        return self.shapes
