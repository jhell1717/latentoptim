import random
from tqdm import tqdm
import matplotlib.pyplot as plt



class GeneticAlgorithm:
    def __init__(
        self,
        population,
        num_generations=50,
        num_parents=4,
        mutation_probability=0.05,
    ):
        self.population = population
        self.num_generations = num_generations
        self.num_parents = num_parents
        self.mutation_probability = mutation_probability
        self.fitness = []

    def generate_offspring(self):
        """_summary_
        """
        new_population = []
        for _ in range(len(self.population.individuals)):
            parent_a, parent_b = random.sample(self.population.parents, 2)
            child = parent_a.crossover(parent_b)
            child.mutate(self.mutation_probability)
            new_population.append(child)

        self.population.individuals = new_population

    def evolutionary_cycle(self):
        """_summary_
        """
        self.population.evaluate()
        self.fitness.append(max(self.population.fitness))
        self.population.select_parents(self.num_parents)
        self.generate_offspring()


    def evolve(self):
        """_summary_
        """
        for _ in tqdm(range(self.num_generations), desc="Evolution"):
            self.evolutionary_cycle()

    def plot_fitness(self):
        """"""
        _, ax = plt.subplots(figsize=(8, 4))
        ax.plot(self.fitness)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness")
        ax.set_title("Fitness Evolution")
        ax.grid(True)
        plt.tight_layout()