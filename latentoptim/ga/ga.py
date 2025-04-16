from .population import Population
import matplotlib.pyplot as plt

class GeneticAlgorithm:
    """_summary_
    """
    def __init__(self,shape_pool,pop_size,generations=100,mutation_rate = 0.05):
        """_summary_

        Args:
            shape_pool (_type_): _description_
            pop_size (_type_): _description_
            generations (int, optional): _description_. Defaults to 100.
            mutation_rate (float, optional): _description_. Defaults to 0.05.
        """
        self.population = shape_pool
        # self.population = Population(shape_pool,pop_size)
        self.generations = generations
        self.mutation_rate = mutation_rate

    def run_ga(self):
        """Run the genetic algorithm for the specified number of generations.
        """
        best_fitness = []
        for gen in range(self.generations):
            self.population.evaluate()
            best = min(self.population.individuals, key=lambda x: x.fitness)
            best_fitness.append(best.fitness)  # Store the actual fitness value
            print(f'Gen {gen} - Best Fitness {best.fitness:.3f}')
            self.population.generate_new_population(self.mutation_rate)
            if gen % 200 == 0:
                best.plot()
        self.plot_results(best_fitness)
        
        

    def plot_results(self,fitness_history):
        plt.plot(fitness_history)
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness (Compactness)')
        plt.title('Fitness over Generations')
        plt.show()


