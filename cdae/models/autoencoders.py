import sys

import torch
from torch import nn
from torch.nn import functional as F
import torchvision.models as models
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ProgressBar
from pytorch_msssim import MS_SSIM

from cdae.models.attention import AttentionConv, SelfAttention


def get_model(config) -> nn.Module:
    """Get model from layer config dictionary."""
    modules = []
    for l in config:
        layer_type = l.pop("type")
        layer = getattr(torch.nn, layer_type)(**l)
        modules.append(layer)
    return nn.Sequential(*modules)


class SimpleAutoencoder(nn.Module):
    """Simple autoencoder."""

    def __init__(self, latent_size: int, layer_config: dict):
        super(SimpleAutoencoder, self).__init__()
        self.latent_size = latent_size
        self.encoder = get_model(layer_config["layers_encoder"])
        self.decoder = get_model(layer_config["layers_decoder"])


    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return x


    def decode(self, x):
        x = x.unsqueeze(2).unsqueeze(3)
        x = self.decoder(x)
        return x


    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x


class LitAutoencoderProgressBar(ProgressBar):
    """ Progress bar for autoencoder.
    """

    def __init__(self):
        super().__init__()  # don't forget this :)
        self.enable = True

    def disable(self):
        self.enable = False

    def get_metrics(self, trainer, model):
        # don't show the version number
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        return super().on_train_epoch_end(trainer, pl_module)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
    


class LightningSimpleAutoencoder(pl.LightningModule):
    """Simple autoencoder."""

    def __init__(self, config):
        super().__init__()

        self.batch_size = config["batch_size"]
        self.latent_size = config["image_latent_size"]
        self.encoder = get_model(config["autoencoder_config"]["layers_encoder"])
        self.decoder = get_model(config["autoencoder_config"]["layers_decoder"])

        self.loss_msssim = MS_SSIM(data_range=1, size_average=True, channel=1)  # TODO: Grayscale flag switch

        self.lr = config["lr"]
        self.weight_decay = config["reg"]

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return x

    def decode(self, x):
        x = x.unsqueeze(2).unsqueeze(3)
        x = self.decoder(x)
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def loss(self, original, reconstructed):
        #return F.mse_loss(original, reconstructed, reduction="mean")
        return 1 - self.loss_msssim(reconstructed, original)

    def training_step(self, batch, batch_idx):
        reconstructed = self.forward(batch[0])
        loss = self.loss(batch[0], reconstructed)
        self.log('loss_train', loss, on_step=False, on_epoch=True, batch_size=batch[0].shape[0], prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        reconstructed = self.forward(batch[0])
        loss = self.loss(batch[0], reconstructed)
        self.log('loss_val', loss, on_step=False, on_epoch=True, batch_size=batch[0].shape[0], prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer


class LatentAttentionAutoencoder(nn.Module):
    """Simple autoencoder."""
    def __init__(self, latent_size: int, layer_config: dict):
        super(LatentAttentionAutoencoder, self).__init__()

        self.latent_size = latent_size
        self.latent_attention = nn.Sequential(
            SelfAttention(),
            nn.BatchNorm1d(num_features=self.latent_size)
        )
        self.encoder = get_model(layer_config["layers_encoder"])
        self.decoder = get_model(layer_config["layers_decoder"])


    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return x


    def decode(self, x):
        x = x.unsqueeze(2).unsqueeze(3)
        x = self.decoder(x)
        return x


    def forward(self, x):
        x = self.encode(x)
        x = self.latent_attention(x)
        x = self.decode(x)
        return x


class CNNAttentionAutoencoder(nn.Module):
    """Simple autoencoder."""
    def __init__(self, latent_size: int, layer_config: dict):
        super(CNNAttentionAutoencoder, self).__init__()

        obs_size = 1
        out_size = 2

        self.attention_conv = nn.Sequential(
            AttentionConv(obs_size,
                          out_size,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          groups=1), nn.BatchNorm2d(out_size))

        self.latent_size = latent_size
        self.encoder = get_model(layer_config["layers_encoder"])
        self.decoder = get_model(layer_config["layers_decoder"])


    def encode(self, x):
        x = self.attention_conv(x)
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return x

    def decode(self, x):
        x = x.unsqueeze(2).unsqueeze(3)
        x = self.decoder(x)
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x


class ResnetAutoencoder(nn.Module):
    """Resnet-backbone autoencoder."""

    def __init__(self, latent_size: int, layer_config: dict, transfer_learning=False):
        super(ResnetAutoencoder, self).__init__()
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.decoder = get_model(layer_config["layers_decoder"])

        # TODO: Test
        if transfer_learning:
            for param in resnet.parameters():
                param.requires_grad = False

        self.fc_l = nn.Sequential(
            nn.Linear(resnet.fc.in_features, 256 * 3 * 3),  # 512 -> 2304
            nn.BatchNorm1d(256 * 3 * 3, momentum=0.01),
            nn.ReLU(),
        )

    def encode(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        return x

    def decode(self, x):
        x = self.fc_l(x)
        x = x.view(-1, 256, 3, 3)
        x = self.decoder(x)
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x