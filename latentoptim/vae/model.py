import torch
import torch.nn as nn


class VAE(nn.Module):
    """_summary
    Args:
        nn (_type_): _description_
    """

    def __init__(self, latent_dim=2):
        """_summary_

        Args:
            latent_dim (int, optional): Size of latent layer dimension. Defaults to 2.
        """
        super(VAE, self).__init__()

        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(200, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 200)
        )

    def reparameterise(self, mu, log_var):
        """_summary_

        Args:
            mu (_type_): Vector of mu for latent dimension attributes.
            log_var (_type_): Vector of variances for latent dimensions attributes.

        Returns:
            _type_: _description_
        """
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        x = self.encoder(x)
        mu, log_var = self.fc_mu(x), self.fc_logvar(x)
        z = self.reparameterise(mu, log_var)
        x_recon = torch.sigmoid(self.decoder(z))
        return x_recon, mu, log_var
