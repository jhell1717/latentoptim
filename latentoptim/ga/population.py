import matplotlib.pyplot as plt
import vae
import data

class Population:

    def __init__(self, individuals):
        self.individuals = individuals
        self.fitness = []
        self.parents = []

    def evaluate(self):
        self.fitness = [vae.Metrics(individual.points).compute_compactness() for individual in self.individuals]
        # self.fitness = [individual.fitness() for individual in self.individuals]

    def select_parents(self, num_parents):
        self.parents = sorted(
            self.individuals, key=lambda x: vae.Metrics(x.points).compute_compactness(), reverse=True
        )[:num_parents]

    def plot(self):
        # fig, axes = plt.subplots(5, 5, figsize=(12, 12))
        for individual in self.individuals:
            individual.plot()
        plt.tight_layout()
        