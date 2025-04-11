import numpy as np


class Individual:
    def __init__(self, genes):
        self.genes = genes
        self._fitness = None

    def evaluate_fitness(self):
        """
        This method should be implemented in subclasses to evaluate fitness.
        """
        raise NotImplementedError

    def fitness(self):
        """
        Returns the fitness of the individual. Computes fitness if it has not
        been computed yet.
        """
        if self._fitness is None:
            self.evaluate_fitness()
        return self._fitness

    def crossover(self, partner):
        """
        Partially mapped crossover (PMX)

        Returns
        -------
        child
        """
        size = len(self.genes)
        child = [-1] * size
        a, b = sorted(np.random.choice(range(size), size=2, replace=False))

        # Copy segment from self to child
        for i in range(a, b):
            child[i] = self.genes[i]

        # Create mapping for the other parent
        mapping = {value: index for index, value in enumerate(child) if value != -1}

        # Fill in the rest from other
        for i in range(size):
            if i < a or i >= b:
                value = partner.genes[i]
                while value in mapping:
                    value = partner.genes[mapping[value]]
                child[i] = value
                mapping[value] = i

        return child

    def mutate(self, mutation_probability):
        """
        Mutation: Swap mutation
        """
        if np.random.rand() < mutation_probability:
            idx1, idx2 = np.random.randint(0, len(self.genes), 2)
            self.genes[idx1], self.genes[idx2] = (
                self.genes[idx2],
                self.genes[idx1],
            )