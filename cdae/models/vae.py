import torch
import torch.nn as nn
import pytorch_lightning as pl

from torch.optim import Adam


class VAE(pl.LightningModule):
    def __init__(self, hparams, net, data_loader):
        super(VAE, self).__init__()
        self.hparams = hparams
        self.net = net
        self.data_loader = data_loader

    def forward(self, x):
        x_out, mu, log_var = self.net.forward(x)
        return x_out, mu, log_var

    def training_step(self, batch, batch_idx):
        x = batch

        # Encode and decode
        x_out, mu, log_var = self.forward(x)

        # KL divergence loss
        kl_loss = (-0.5 *
                   (1 + log_var - mu**2 - torch.exp(log_var)).sum(dim=1)).mean(
                       dim=0)

        # Reconstruction loss
        recon_loss_criterion = nn.MSELoss()
        recon_loss = recon_loss_criterion(x, x_out)

        # Total loss
        loss = self.hparams.alpha * recon_loss + self.hparams.beta * kl_loss

        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch

        # Encode and decode
        x_out, mu, log_var = self.forward(x)

        # Loss
        kl_loss = (-0.5 *
                   (1 + log_var - mu**2 - torch.exp(log_var)).sum(dim=1)).mean(
                       dim=0)
        recon_loss_criterion = nn.MSELoss()
        recon_loss = recon_loss_criterion(x, x_out)
        loss = self.hparams.alpha * recon_loss + self.hparams.beta * kl_loss

        self.log('val_kl_loss', kl_loss, on_step=False, on_epoch=True)
        self.log('val_recon_loss', recon_loss, on_step=False, on_epoch=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        return x_out, loss

    def train_dataloader(self):
        return self.data_loader['train_dataloader']

    def val_dataloader(self):
        return self.data_loader['val_dataloader']

    def test_dataloader(self):
        return self.data_loader['test_dataloader']

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)

    def scale_image(self, img):
        out = (img + 1) / 2
        return out
