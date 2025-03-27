import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull

class Shape:
    def __init__(self, points, n_points=None, normalise=True,rotate=False,skew = True):
        """
        Base class for shapes, with optional resampling.

        Parameters
        ----------
        points : ndarray of shape (N, 2)
            The (x, y) coordinates of the shape's points.

        n_points : int, optional
            Number of points to resample the shape to. If None, no resampling
            is performed.
        """

        if n_points is not None:
            self.points = self._resample(points, n_points)

        if skew and np.random.randn() < 0.3:
            self.points = self.skew_shape()
        
        if rotate:
            self.points = self.rotate_shape()

        if normalise:
            self.points = self._normalise_shape(self.points)


    @staticmethod
    def _resample(points, n_points):
        """
        Resample the shape to have a fixed number of points.

        Parameters
        ----------
        points : ndarray of shape (N, 2)
            Original points of the shape.

        n_points : int
            Desired number of points.

        Returns
        -------
        ndarray of shape  n_points, 2)
            Resampled points.
        """
        closed_points = np.vstack([points, points[0]])
        distances = np.cumsum(np.linalg.norm(
            np.diff(closed_points, axis=0), axis=1))
        distances = np.insert(distances, 0, 0)
        interp_func = interp1d(distances, closed_points, axis=0, kind="linear")
        uniform_distances = np.linspace(0, distances[-1], n_points)
        return interp_func(uniform_distances)

    def _normalise_shape(self, points):
        """
        Normalises the shape points to the [0, 1] range.

        Returns
        -------
        numpy.ndarray
            Normalised points of the shape.
        """
        min_val = points.min()
        max_val = points.max()
        return (points - min_val) / (max_val - min_val)

    def plot(self, ax=None):
        """
        Plot the shape.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Matplotlib axis object. If None, a new figure is created.
        """
        if ax is None:
            _, ax = plt.subplots()
        closed_points = np.vstack([self.points, self.points[0]])
        ax.plot(closed_points[:, 0], closed_points[:, 1], "-o", label="Shape",markersize=3)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

    def rotate_shape(self,max_rotation = 360):
        angle = np.random.uniform(0, max_rotation)  # Pick a random rotation angle
        theta = np.radians(angle)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])
        
        centroid = np.mean(self.points,axis=0)
        return (self.points - centroid) @ rotation_matrix.T + centroid
    
    def skew_shape(self, max_shear=0.8):
        """_summary_

        Args:
            max_shear (float, optional): _description_. Defaults to 0.5.

        Returns:
            _type_: _description_
        """
        shear_x = np.random.uniform(-max_shear, max_shear)  # Shear factor for x
        shear_y = np.random.uniform(-max_shear, max_shear)  # Shear factor for y

        shear_matrix = np.array([[1, shear_x],
                                [shear_y, 1]])

        return  self.points @ shear_matrix.T  # Apply transformation


class Square(Shape):

    def __init__(self, side_length=1, n_points=50):
        if side_length is None:
            side_length = np.random.uniform(0.1, 1)
        points = np.array(
            [
                [-side_length / 2, -side_length / 2],
                [side_length / 2, -side_length / 2],
                [side_length / 2, side_length / 2],
                [-side_length / 2, side_length / 2],
            ]
        )
        super().__init__(points, n_points)




class Circle(Shape):
    def __init__(self, radius=None, n_points=100):
        """
        A circle defined by its radius and number of points.

        Parameters
        ----------
        radius : float
            Radius of the circle.

        n_points : int
            Number of points to approximate the circle.
        """
        if radius is None:
            radius = np.random.uniform(0.1, 1)
        theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        points = np.column_stack(
            (radius * np.cos(theta), radius * np.sin(theta)))
        super().__init__(points, n_points)


class Triangle(Shape):

    def __init__(self, n_points=100, vertices=None):
        """
        A triangle defined by three vertices, resampled to a fixed number of points.

        Parameters
        ----------
        n_points : int
            Number of points to resample the triangle to.

        vertices : ndarray of shape (3, 2), optional
            Coordinates of the triangle's vertices. If None, a random triangle
            is generated.
        """
        if vertices is None:
            vertices = self._generate_valid_triangle()
        super().__init__(vertices, n_points)

    @staticmethod
    def _generate_valid_triangle():
        """Generate a random valid triangle."""
        while True:
            points = np.random.uniform(-1, 1, size=(3, 2))
            if Triangle._is_valid_triangle(points):
                return points

    @staticmethod
    def _is_valid_triangle(points):
        """
        Check if the given points form a valid triangle.

        Parameters
        ----------
        points : ndarray of shape (3, 2)
            Points to check.

        Returns
        -------
        bool
            True if the points form a valid triangle, False otherwise.
        """
        v1 = points[1] - points[0]
        v2 = points[2] - points[0]
        return np.abs(np.cross(v1, v2)) > 1e-6


class Rectangle(Shape):
    def __init__(self, width=None, height=None, n_points=100):
        width = width or np.random.uniform(0.5, 1.5)
        height = height or np.random.uniform(0.5, 1.5)
        points = np.array([[-width/2, -height/2], [width/2, -height/2],
                           [width/2, height/2], [-width/2, height/2]])
        super().__init__(points, n_points)


class Diamond(Shape):

    def __init__(self, width=None, height=None, n_points=100):
        if width is None:
            width = np.random.uniform(0.1, 0.5)
        if height is None:
            height = np.random.uniform(0.5, 1)
        points = np.array(
            [[0, -height / 2], [width / 2, 0], [0, height / 2], [-width / 2, 0]]
        )
        super().__init__(points, n_points)


class Heart(Shape):

    def __init__(self, scale=1, n_points=100):
        t = np.linspace(0, 2 * np.pi, n_points)
        x = scale * (16 * np.sin(t) ** 3)
        y = scale * (
            13 * np.cos(t) - 5 * np.cos(2 * t) - 2 *
            np.cos(3 * t) - np.cos(4 * t)
        )
        points = np.column_stack((x, y))
        super().__init__(points, n_points)


class Oval(Shape):

    def __init__(self, major_axis=None, minor_axis=None, n_points=100):
        if major_axis is None:
            major_axis = np.random.uniform(1, 2)
        if minor_axis is None:
            minor_axis = np.random.uniform(0.5, 1)
        theta = np.linspace(0, 2 * np.pi, n_points)
        x = major_axis / 2 * np.cos(theta)
        y = minor_axis / 2 * np.sin(theta)
        points = np.column_stack((x, y))
        super().__init__(points, n_points)


class Pentagon(Shape):

    def __init__(self, radius=None, n_points=100):
        if radius is None:
            radius = np.random.uniform(0.1, 1)
        theta = np.linspace(0, 2 * np.pi, 6, endpoint=True)
        points = np.column_stack(
            (radius * np.cos(theta), radius * np.sin(theta)))[:-1]
        super().__init__(points, n_points)


class Star(Shape):

    def __init__(self, n_arms=None, outer_radius=None, inner_radius=None, n_points=100):
        """
        A star shape with alternating inner and outer points.

        Parameters
        ----------
        n_arms : int
            Number of arms on the star.

        outer_radius : float
            Radius of the outer points.

        n_points : int
            Number of points to resample the star shape to.
        """
        if n_arms is None:
            n_arms = np.random.randint(5, 10)
        if outer_radius is None:
            outer_radius = np.random.uniform(0.1, 1)
            inner_radius = outer_radius / np.random.uniform(1.5, 4)

        theta = np.linspace(0, 2 * np.pi, n_arms * 2, endpoint=False)
        radii = np.array(
            [outer_radius if i % 2 == 0 else inner_radius for i in range(len(theta))]
        )
        points = np.column_stack((radii * np.cos(theta), radii * np.sin(theta)))
        super().__init__(points, n_points)


