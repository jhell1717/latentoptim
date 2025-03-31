import os
import csv

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

# Use LaTeX-style formatting
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12
})

plt.rcParams["figure.dpi"] = 300  # High resolution

plt.rcParams["font.family"] = "serif"  # Use serif font
plt.rcParams["axes.linewidth"] = 1.5  # Thicker axis lines
plt.rcParams["xtick.direction"] = "in"  # Ticks inside the plot
plt.rcParams["ytick.direction"] = "in"


class ModelRecord:
    def __init__(self, base_dir, model_name, trained_data, batch_size):
        self.base_dir = base_dir
        self.model_name = model_name
        self.trained_data = trained_data
        self.batch_size = batch_size

        self.model_dir = self._get_unique_model_dir()
        os.makedirs(self.model_dir, exist_ok=False)

    def _get_unique_model_dir(self):
        """_summary_

        Args:
            model_name (_type_): _description_

        Returns:
            _type_: _description_
        """

        if self.model_name:
            return os.path.join(self.base_dir, self.model_name)

    def save_checkpoint(self, epoch):
        """_summary_

        Args:
            epoch (_type_): _description_
        """
        checkpoint_path = os.path.join(
            self.model_dir, f'vae_epoch_{epoch}_mse{self.epoch_loss:.1f}.pt')

        torch.save(self.model, checkpoint_path)

    def log_results(self, epochs, model_loc):
        csv_loc = os.path.join(self.base_dir, 'test_log.csv')
        file_exists = os.path.isfile(csv_loc)

        header = ['Model Name', 'Save Directory', 'Final Train Loss', 'Epochs',
                  'Batch Size','Model Link', 'Trained Data', 'Learning Rate']
        row = [self.model_name, self.model_dir, self.epoch_loss, epochs,self.batch_size,model_loc,self.trained_data,self.lr]

        with open(csv_loc,mode='a',newline="") as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(header)
            writer.writerow(row)


class Trainer(ModelRecord):
    """_summary_
    """

    def __init__(self, data, model, base_dir, trained_data, model_name, batch_size, val_data=None, lr=1e-3):
        """_summary_

        Args:
            data (_type_): _description_
            model (_type_): _description_
            lr (_type_, optional): _description_. Defaults to 1e-3.
            batch_size (int, optional): _description_. Defaults to 128.
        """
        super().__init__(base_dir, model_name, trained_data, batch_size)
        self.lr = lr
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.optimiser = Adam(self.model.parameters(), lr=self.lr)
        self.epoch_loss = None
        self.val_loss = None

        self.dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
        self.val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False) if val_data is not None else None

        os.makedirs(self.model_dir, exist_ok=True)

        self.loss_history = []
        self.val_loss_history = []

    def train(self):
        """Training loop for one epoch"""
        self.epoch_loss = 0
        self.model.train()
        for batch in self.dataloader:
            x = batch[0].to(self.device)
            self.optimiser.zero_grad()
            x_recon, mu, logvar = self.model(x)
            loss = Loss(x, x_recon, mu, logvar).compute_loss_mse()
            loss.backward()
            self.optimiser.step()
            self.epoch_loss += loss.item()

    def validate(self):
        """Validation loop"""
        if self.val_dataloader is None:
            return None
            
        self.val_loss = 0
        self.model.eval()
        with torch.no_grad():
            for batch in self.val_dataloader:
                x = batch[0].to(self.device)
                x_recon, mu, logvar = self.model(x)
                loss = Loss(x, x_recon, mu, logvar).compute_loss_mse()
                self.val_loss += loss.item()
        return self.val_loss

    def train_model(self, model_name, epochs=10, checkpoint_interval=100):
        """Train the model for specified number of epochs"""
        self.model.train()
        for epoch in range(epochs):
            self.train()
            val_loss = self.validate()

            self.loss_history.append(self.epoch_loss)
            if val_loss is not None:
                self.val_loss_history.append(val_loss)
            
            print(f"Epoch {epoch+1}, Train Loss: {self.epoch_loss:.2f}", end="")
            if val_loss is not None:
                print(f", Val Loss: {val_loss:.2f}")
            else:
                print()

            if epoch % checkpoint_interval == 0:
                self.save_checkpoint(epoch)

        save_path = os.path.join(
            self.model_dir, f'vae_epoch_{epochs}_mse{self.epoch_loss:.1f}.pt')
        torch.save(self.model, save_path)
        self.log_results(epochs, save_path)

        self._plot_loss()

    def _plot_loss(self):
        """Plots and saves the loss curves."""
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(self.loss_history) + 1),
                 self.loss_history, label='Train Loss', color='b')
        if self.val_loss_history:
            plt.plot(range(1, len(self.val_loss_history) + 1),
                     self.val_loss_history, label='Validation Loss', color='r')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss Over Time")
        plt.legend()
        plt.grid()

        # Save plot in the same model directory
        loss_plot_path = os.path.join(self.model_dir, "loss_plot.png")
        plt.savefig(loss_plot_path)
        plt.close()
        print(f"Loss plot saved to {loss_plot_path}")


class Loss:
    """_summary_
    """

    def __init__(self, x, x_recon, mu, logvar, beta=10):
        """_summary_

        Args:
            x (_type_): _description_
            x_recon (_type_): _description_
            mu (_type_): _description_
            logvar (_type_): _description_
        """
        self.beta = beta
        self.x = x
        self.x_recon = x_recon
        self.mu = mu
        self.logvar = logvar

    def compute_loss_mse(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        mse = nn.functional.mse_loss(self.x_recon, self.x, reduction='sum')
        # recon_loss = nn.MSELoss(reduction='sum')(self.x_recon, self.x)
        kl_loss = -0.5 * torch.sum(1+self.logvar -
                                   self.mu.pow(2) - self.logvar.exp())
        return mse + self.beta * kl_loss

    def compute_loss_bce(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        recon_loss = nn.functional.binary_cross_entropy(
            self.x_recon, self.x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1+self.logvar -
                                   self.mu.pow(2) - self.logvar.exp())
        return recon_loss + kl_loss
