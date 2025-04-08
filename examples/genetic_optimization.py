import torch
import matplotlib.pyplot as plt
import numpy as np
from latentoptim.vae.model import VAE
from latentoptim.vae.genetic_optimizer import GeneticOptimizer
from latentoptim.data.generator import Generator
from latentoptim.vae.trainer import Trainer

def main():
    # Generate training data
    generator = Generator(resolution=100, num_shapes=1000)
    shapes = generator.generate_shapes()
    
    # Create and train VAE
    input_size = 200  # 100 points * 2 coordinates
    latent_dim = 8
    vae = VAE(input_size=input_size, latent_dim=latent_dim)
    
    # Train the VAE
    trainer = Trainer(
        data=shapes,
        model=vae,
        base_dir="models",
        trained_data="shapes",
        model_name="vae_model",
        batch_size=32,
        lr=1e-3
    )
    trainer.train_model(epochs=100)
    
    # Initialize genetic optimizer
    optimizer = GeneticOptimizer(
        vae_model=vae,
        population_size=50,
        mutation_rate=0.1,
        mutation_scale=0.2,
        elite_size=5
    )
    
    # Run optimization
    best_individual, fitness_history = optimizer.optimize(
        generations=100,
        verbose=True
    )
    
    # Decode and plot the best shape
    with torch.no_grad():
        best_shape = vae.decoder(best_individual).numpy().reshape(-1, 2)
    
    # Plot the optimization progress
    plt.figure(figsize=(10, 5))
    plt.plot(fitness_history)
    plt.xlabel('Generation')
    plt.ylabel('Fitness (1/Compactness)')
    plt.title('Genetic Algorithm Optimization Progress')
    plt.grid(True)
    plt.savefig('optimization_progress.png')
    plt.close()
    
    # Plot the best shape
    plt.figure(figsize=(6, 6))
    plt.plot(best_shape[:, 0], best_shape[:, 1], 'b-')
    plt.plot([best_shape[-1, 0], best_shape[0, 0]], 
             [best_shape[-1, 1], best_shape[0, 1]], 'b-')
    plt.axis('equal')
    plt.title('Best Shape Found')
    plt.savefig('best_shape.png')
    plt.close()

if __name__ == "__main__":
    main() 