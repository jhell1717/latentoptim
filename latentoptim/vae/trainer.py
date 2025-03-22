
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader


class Trainer:
    """_summary_
    """

    def __init__(self, data, model, lr=1e-3, batch_size=128,loss = 'mse'):
        """_summary_

        Args:
            data (_type_): _description_
            model (_type_): _description_
            lr (_type_, optional): _description_. Defaults to 1e-3.
            batch_size (int, optional): _description_. Defaults to 128.
        """
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.optimiser = Adam(self.model.parameters(), lr=lr)
        self.epoch_loss = None
        self.dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

    def train(self):
        """_summary_
        """
        self.epoch_loss = 0
        for batch in self.dataloader:
            x = batch[0].to(self.device)
            self.optimiser.zero_grad()
            x_recon, mu, logvar = self.model(x)
            loss = Loss(x, x_recon, mu, logvar).compute_loss_mse()
            loss.backward()
            self.optimiser.step()
            self.epoch_loss += loss.item()

    def train_model(self, epochs=10):
        """_summary_

        Args:
            epochs (int, optional): _description_. Defaults to 10.
        """
        self.model.train()
        for epoch in range(epochs):
            self.train()
            print(f"Epoch {epoch+1}, Loss: {self.epoch_loss}")


class Loss:
    """_summary_
    """

    def __init__(self, x, x_recon, mu, logvar):
        """_summary_

        Args:
            x (_type_): _description_
            x_recon (_type_): _description_
            mu (_type_): _description_
            logvar (_type_): _description_
        """
        self.x = x
        self.x_recon = x_recon
        self.mu = mu
        self.logvar = logvar

    def compute_loss_mse(self,beta=1.0):
        """_summary_

        Returns:
            _type_: _description_
        """
        mse = nn.functional.mse_loss(self.x_recon,self.x,reduction='sum')
        # recon_loss = nn.MSELoss(reduction='sum')(self.x_recon, self.x)
        kl_loss = -0.5 * torch.sum(1+self.logvar - self.mu.pow(2) - self.logvar.exp())
        return mse + beta * kl_loss
    
    def compute_loss_bce(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        recon_loss = nn.functional.binary_cross_entropy(self.x_recon,self.x,reduction='sum')
        kl_loss = -0.5 * torch.sum(1+self.logvar - self.mu.pow(2) - self.logvar.exp())
        return recon_loss + kl_loss
