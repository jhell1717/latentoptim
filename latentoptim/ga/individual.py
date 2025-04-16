import numpy as np
import copy
import vae

class Individual:
    def __init__(self,points):
        """_summary_

        Args:
            points (_type_): _description_
        """
        self.points = copy.deepcopy(points)
        self.fitness = vae.Metrics(self.points).compute_compactness()

    def evaluate_fitness(self):
        """_summary_
        """
        self.fitness = vae.Metrics(self.points).compute_compactness()

    def mutate(self,mutation_rate=0.05,noise_std=0.01):
        """Mutate the individual's points while maintaining shape closure.

        Args:
            mutation_rate (float, optional): Probability of mutation for each point. Defaults to 0.05.
            noise_std (float, optional): Standard deviation of the Gaussian noise. Defaults to 0.01.

        Returns:
            Individual: A new mutated individual.
        """
        mutated = self.points.copy()
        # Don't mutate the last point (it should stay identical to the first point)
        mask = np.random.rand(mutated.shape[0]-1, mutated.shape[1]) < mutation_rate
        mutated[:-1] += mask * np.random.normal(0, noise_std, (mutated.shape[0]-1, mutated.shape[1]))
        # Ensure the last point matches the first point
        mutated[-1] = mutated[0]
        return Individual(mutated)
    
    def crossover(self,other):
        """Perform crossover between two individuals to create a child while maintaining shape closure.

        Args:
            other (Individual): The other parent individual.

        Returns:
            Individual: The child individual.
        """
        parent_1 = self.points
        parent_2 = other.points
        size = len(parent_1)

        # Choose a random crossover point (excluding the last point to maintain closure)
        cx_point = np.random.randint(1, size-2)
        
        # Create child by combining segments from both parents
        child = np.zeros_like(parent_1)
        child[:cx_point] = parent_1[:cx_point]
        child[cx_point:-1] = parent_2[cx_point:-1]  # Don't include the last point yet
        # Ensure the last point matches the first point
        child[-1] = child[0]
        
        return Individual(child)
    
    def plot(self):
        import matplotlib.pyplot as plt
        x,y = self.points[:,0],self.points[:,1]
        plt.plot(x,y)
        plt.axis('equal')
        plt.show()

