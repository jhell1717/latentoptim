import vae
import numpy as np
from .individual import Individual


class Population:
    def __init__(self,shape_pool,size):
        self.raw_shapes = shape_pool.get_n_random_shapes(size)
        self.individuals = [Individual(shape.points)for shape in self.raw_shapes]

    @staticmethod
    def _check_intersection(p1, p2, p3, p4):
        """Check if two line segments (p1-p2 and p3-p4) intersect.
        
        Args:
            p1, p2, p3, p4 (numpy.ndarray): Points defining the line segments.
            
        Returns:
            bool: True if the segments intersect, False otherwise.
        """
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
        
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

    @staticmethod
    def _has_self_intersections(points):
        """Check if a shape has any self-intersections.
        
        Args:
            points (numpy.ndarray): Array of shape points.
            
        Returns:
            bool: True if the shape has self-intersections, False otherwise.
        """
        n = len(points)
        for i in range(n-1):
            for j in range(i+2, n-1):
                if Population._check_intersection(points[i], points[i+1], points[j], points[j+1]):
                    return True
        return False

    @staticmethod
    def generate_random_points(n_points=200):
        """Generate random points that form a valid closed shape.
        
        Args:
            n_points (int): Number of points in the shape. Defaults to 200.
            max_attempts (int): Maximum number of attempts to generate a valid shape. Defaults to 100.
            
        Returns:
            numpy.ndarray: Array of shape (n_points, 2) containing the points.
        """
        # Generate random angles for n_points-1 points (to leave room for closing point)
        angles = np.sort(np.random.uniform(0, 2*np.pi, n_points-1))
        # Add the first angle again at the end to close the shape
        angles = np.append(angles, angles[0])
        
        # Generate random radii for each angle (use same radius for first/last point)
        radii = np.random.uniform(0.3, 1.0, n_points-1)
        radii = np.append(radii, radii[0])  # Use same radius for closing point
        
        # Convert polar coordinates to cartesian
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        
        # Center and normalize the shape
        x = (x - x.min()) / (x.max() - x.min())
        y = (y - y.min()) / (y.max() - y.min())
        
        points = np.column_stack((x, y))
        
        # Verify first and last points are identical
        assert np.allclose(points[0], points[-1]), "Shape is not closed"
        
        return points

    @classmethod
    def create_random_population(cls, size, n_points=100):
        """Create a population of random shapes.
        
        Args:
            size (int): Number of individuals in the population.
            n_points (int): Number of points per shape. Defaults to 200.
            
        Returns:
            Population: A new population with random shapes.
        """
        # Create a dummy shape pool
        class DummyShapePool:
            def get_n_random_shapes(self, n):
                return [type('Shape', (), {'points': cls.generate_random_points(n_points)}) for _ in range(n)]
        
        return cls(DummyShapePool(), size)

    def evaluate(self):
        for ind in self.individuals:
            ind.evaluate_fitness()

    def select_parents(self,k=900):
        selected = []
        for _ in range(len(self.individuals)):
            tournament = np.random.choice(self.individuals,k)
            selected.append(min(tournament,key=lambda x: x.fitness))
        return selected
    
    def generate_new_population(self,mutation_rate = 0.05):
        """Generate a new population through selection, crossover and mutation while maintaining population size.

        Args:
            mutation_rate (float, optional): Probability of mutation. Defaults to 0.05.
        """
        parents = self.select_parents()
        children = []
        target_size = len(self.individuals)
        
        while len(children) < target_size:
            # Randomly select two parents
            parent1, parent2 = np.random.choice(parents, 2, replace=False)
            child = parent1.crossover(parent2)
            mutated_child = child.mutate(mutation_rate)
            children.append(mutated_child)
            
        self.individuals = children
            
