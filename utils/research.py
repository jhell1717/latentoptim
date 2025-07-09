import os
import itertools
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import matplotlib.cm as cm
import matplotlib.colors as colors


def plot_latent_kde(shapes_path, model, num_z=4, max_samples=1000,
                    save_path=None, save_dir='latent_space_plots', model_name=None):
    """Plots KDE of latent space and optionally saves to disk."""

    # Load shapes
    with open(shapes_path, 'rb') as f:
        shapes = pickle.load(f)

    # Convert to tensors
    sh = [torch.tensor(shape.points, dtype=torch.float32).view(-1) for shape in shapes]

    # Get latent vectors
    z_values = []
    for i in sh:
        with torch.no_grad():
            x = model.encoder(i)
            mu, logvar = model.fc_mu(x), model.fc_logvar(x)
            z = model.reparameterise(mu, logvar)
            z_values.append(z.detach().numpy())

    z_values = np.array(z_values)

    # Plot
    plt.figure(figsize=(min(6 + num_z, 10), 4))
    for i in range(min(num_z, z_values.shape[1])):
        sns.kdeplot(z_values[:, i], label=f'Latent Dimension {i + 1}', fill=False)

    plt.xlabel('Latent Dimension Value')
    plt.ylabel('Density')
    plt.title(f'Latent Space KDE Plot (First {num_z} Dimensions)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Determine where to save the figure
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        # Default save path follows the structure used in plot_all_latent_combinations
        latent_dim = num_z
        model_folder = os.path.join(save_dir, f"latent_dim_{latent_dim}")
        os.makedirs(model_folder, exist_ok=True)
        filename = f"kde_latent_dim_{latent_dim}"
        if model_name:
            filename += f"_{model_name}"
        filename += ".png"
        full_path = os.path.join(model_folder, filename)
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        plt.close()


def plot_latent_space_with_compactness(model, vae_metrics, latent_dim,
                                       fixed_dims=None,  # dict: {index: value}
                                       # tuple: indices of z to vary
                                       grid_dims=(0, 1),
                                       grid_size=50, z_min=-3.0, z_max=3.0,
                                       resolution=200,
                                       save_path=None):
    assert latent_dim >= 2, "Latent dimension must be at least 2 for this visualization."
    assert fixed_dims is not None, "You must specify which dimensions to fix (e.g., {2: 0.0})."
    assert len(
        grid_dims) == 2, "You must specify exactly two latent dimensions to vary."

    z1 = np.linspace(z_min, z_max, grid_size)
    z2 = np.linspace(z_min, z_max, grid_size)
    z_grid = np.array(np.meshgrid(z1, z2)).T.reshape(-1, 2)

    cmap = cm.jet
    compactness_values = []

    for z_pair in z_grid:
        z_full = np.zeros(latent_dim)
        z_full[grid_dims[0]] = z_pair[0]
        z_full[grid_dims[1]] = z_pair[1]
        for dim_idx, val in fixed_dims.items():
            z_full[dim_idx] = val

        with torch.no_grad():
            z_tensor = torch.tensor(z_full, dtype=torch.float32).unsqueeze(0)
            decoded_shape = model.decoder(
                z_tensor).detach().numpy().reshape(resolution, 2)
            compactness = vae_metrics(decoded_shape).compute_compactness()
            compactness_values.append(compactness)

    compactness_values = np.array(compactness_values)
    compactness_min = np.percentile(compactness_values, 5)
    compactness_max = np.percentile(compactness_values, 95)
    norm = colors.LogNorm(vmin=compactness_min, vmax=compactness_max)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xlim(z_min, z_max)
    ax.set_ylim(z_min, z_max)
    ax.set_aspect('equal')
    ax.set_xlabel(f'$z_{{{grid_dims[0]+1}}}$', fontsize=10)
    ax.set_ylabel(f'$z_{{{grid_dims[1]+1}}}$', fontsize=10)
    fixed_str = ", ".join(
        [f"$z_{{{k+1}}}={v}$" for k, v in fixed_dims.items()])
    ax.set_title(f'Latent Space Slice\n{fixed_str}\nwith Compactness Overlay')

    shape_scale = 0.1
    for idx, z_pair in enumerate(z_grid):
        z_full = np.zeros(latent_dim)
        z_full[grid_dims[0]] = z_pair[0]
        z_full[grid_dims[1]] = z_pair[1]
        for dim_idx, val in fixed_dims.items():
            z_full[dim_idx] = val

        z_tensor = torch.tensor(z_full, dtype=torch.float32).unsqueeze(0)
        decoded_shape = model.decoder(
            z_tensor).detach().numpy().reshape(resolution, 2)
        compactness = compactness_values[idx]
        decoded_shape *= shape_scale
        color_value = cmap(norm(compactness))
        ax.fill(decoded_shape[:, 0] + z_pair[0], decoded_shape[:, 1] + z_pair[1],
                color=color_value, alpha=0.3, edgecolor='black', linewidth=0.3)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(
        r"Compactness ($\frac{\text{Perimeter}^2}{\text{Area}}$)", fontsize=10)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_all_latent_combinations(models, latent_dims, vae_metrics,
                                 output_dir="latent_space_plots",
                                 z_min=-3.0, z_max=3.0, grid_size=50, resolution=200,
                                 shapes_path=None):
    

    for model_idx, (model, latent_dim) in enumerate(zip(models, latent_dims)):
        print(f"\n Plotting for latent dimension: {latent_dim}D")

        all_indices = list(range(latent_dim))
        latent_pairs = list(itertools.combinations(all_indices, 2))
        model_folder = os.path.join(output_dir, f"latent_dim_{latent_dim}")
        os.makedirs(model_folder, exist_ok=True)

        # Plot KDE and save it in the same folder
        if shapes_path:
            print("  ➤ Generating KDE plot...")
            plot_latent_kde(
                shapes_path=shapes_path,
                model=model,
                num_z=latent_dim,
                save_dir=output_dir,
                model_name=f"model{model_idx+1}"
            )

        # Plot 2D latent combinations with compactness overlay
        for grid_dims in latent_pairs:
            fixed_dims = {i: 1.0 for i in all_indices if i not in grid_dims}
            fixed_str = "_".join(
                [f"z{i+1}={v}" for i, v in fixed_dims.items()])
            filename = f"varying_z{grid_dims[0]+1}_z{grid_dims[1]+1}_fixed_{fixed_str}.png"
            save_path = os.path.join(model_folder, filename)

            print(f"  ➤ Saving: {filename}")
            plot_latent_space_with_compactness(
                model=model,
                vae_metrics=vae_metrics,
                latent_dim=latent_dim,
                fixed_dims=fixed_dims,
                grid_dims=grid_dims,
                grid_size=grid_size,
                z_min=z_min,
                z_max=z_max,
                resolution=resolution,
                save_path=save_path
            )
