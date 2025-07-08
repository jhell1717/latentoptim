import torch
import torch.nn as nn


class VAE(nn.Module):
    """_summary
    Args:
        nn (_type_): _description_
    """

    def __init__(self,input_size, latent_dim=2):
        """_summary_

        Args:
            latent_dim (int, optional): Size of latent layer dimension. Defaults to 2.
        """
        super(VAE, self).__init__()

        self.latent_dim = latent_dim
        self.input_size = input_size

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_size)
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
        x_recon = self.decoder(z)
        return x_recon, mu, log_var
