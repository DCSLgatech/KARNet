
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_msssim import MS_SSIM


def get_model(config) -> nn.Module:
    """Get model from layer config dictionary."""
    modules = []
    for l in config:
        layer_type = l.pop("type")
        layer = getattr(torch.nn, layer_type)(**l)
        modules.append(layer)
    return nn.Sequential(*modules)


class SimpleLatentPredictionNet(nn.Module):
    """Latent prediction net - RNN prediction in latent space."""

    def __init__(
        self,
        autoencoder: torch.nn.Module,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0,
        device: str = "cuda",   # TODO: Change
        rnn_arch: str = "lstm",
        h_0_latent = True
    ):
        super(SimpleLatentPredictionNet, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device
        self.rnn_arch = rnn_arch
        self.h_0_latent = h_0_latent
        # Autoencoder Part
        self.autoencoder = autoencoder

        if rnn_arch == "lstm":
            # RNN Part
            self.rnn = nn.LSTM(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=True,  # Need to do batch_first on this one
            )
        elif rnn_arch == "gru":
            self.rnn = nn.GRU(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=True,
            )
            pass


    def encode(self, x):
        x = self.autoencoder.encode(x)
        return x

    def decode(self, x):
        x = self.autoencoder.decode(x)
        return x

    def forward(self, x_in):
        """Forward function.

        TODO: Optimize reshapes.

        Args:
            x_in (torch.Tensor): input tensor of shape (batch_size, no_seq+1, C, H, W)
                Contains images at t_0 .. t_n, and target image at t_{n+1}.
                Example: (16, 6, 3, 224, 224), where sequence length is 5 and one image is predicted.

        Returns:
            x_out (torch.Tensor): tensor containing predicted reconstructed images (batch, no_seq, c, h, w).
                Contains images at t_1 .. t_{n+1}. Example; (16, 5, 3, 224, 224).
            x_out_ae (torch.Tensor): tensor containing reconstructed original images (batch, no_seq+1, c, h, w).
            x_out_lat (torch.Tensor): tensor containing predicted latent states (batch, seq, hidden_size).
            x_in_lat (torch.Tensor):  tensor containing latent representation of x_in.
        """

        xs = x_in.shape  # (batch, (seq+1), c, h, w)
        x_in = x_in.view(-1, xs[2], xs[3], xs[4])  # (batch*(seq+1),  c, h, w)

        # Encode all image sequence (n_batch x (n_seq+1)) -> latent representation
        x_in_lat = self.encode(x_in)  # (batch*(seq+1),  input_size)
        x_out_ae = self.decode(x_in_lat)  # Decode again
        x_in_lat = x_in_lat.view(xs[0], xs[1], -1)  # (batch, seq, input_size)

        # Initialize hidden and cell states for LSTM
        # (self.num_layers, batch_size, self.hidden_size)
        #h_0 = torch.zeros(self.num_layers, xs[0], self.hidden_size).to(self.device)
        # TODO: Number of layers
        if self.h_0_latent:
            h_0 = torch.unsqueeze(x_in_lat[:,0, :],0).contiguous()
        else:
            h_0 = torch.zeros(self.num_layers, xs[0], self.hidden_size).to(self.device)

        if self.rnn_arch == "lstm":
            c_0 = torch.zeros(self.num_layers, xs[0], self.hidden_size).to(self.device)
            # Predict based on (seq-1) frames
            x_out_lat, (h_0, c_0) = self.rnn(x_in_lat[:, :-1, :], (h_0, c_0))  # (batch, seq, hidden_size)
        elif self.rnn_arch == "gru":
            x_out_lat, h_0 = self.rnn(x_in_lat[:, :-1, :], h_0)  # (batch, seq, hidden_size)

        # TODO: Change input size to other when hidden size is different (add fc layers)/
        x_out = self.decode(x_out_lat.reshape(-1, self.hidden_size))
        x_out = x_out.view(xs[0], xs[1] - 1, xs[2], xs[3], xs[4])

        return x_out, x_out_ae, x_out_lat, x_in_lat


class LightningSimpleLatentPredictionNet(pl.LightningModule):
    """ Lightning module for latent sensor prediction network.
    """

    def __init__(self, config):
        super().__init__()

        self.input_size = config["image_latent_size"]
        self.hidden_size = config["image_latent_size"]
        self.num_layers = config["lstm_n_layers"]
        self.dropout = config["dropout"]
        #self.device = config["device"]
        self.rnn_arch = config["rnn_arch"]
        self.h_0_latent = config["h_0_latent"]  # Whether to initiate h_0 with first 
                                                # latent encoding or not
        self.no_seq = config["no_seq"]
        self.loss_last_step = config["last_image"]

        # Autoencoder Part
        self.autoencoder = torch.load(config["autoencoder_resume"])
        self.loss_msssim = MS_SSIM(data_range=1, size_average=True, channel=1)

        self.lr = config["lr"]
        self.weight_decay = config["reg"]

        if self.rnn_arch == "lstm":
            # RNN Part
            self.rnn = nn.LSTM(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=True,  # Need to do batch_first on this one
            )
        elif self.rnn_arch == "gru":
            self.rnn = nn.GRU(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=True,
            )
        else:
            raise NotImplementedError


    def encode(self, x):
        x = self.autoencoder.encode(x)
        return x

    def decode(self, x):
        x = self.autoencoder.decode(x)
        return x

    def forward(self, x_in):
        """Forward function.

        TODO: Optimize reshapes.

        Args:
            x_in (torch.Tensor): input tensor of shape (batch_size, no_seq+1, C, H, W)
                Contains images at t_0 .. t_n, and target image at t_{n+1}.
                Example: (16, 6, 3, 224, 224), where sequence length is 5 and one image is predicted.

        Returns:
            x_out (torch.Tensor): tensor containing predicted reconstructed images (batch, no_seq, c, h, w).
                Contains images at t_1 .. t_{n+1}. Example; (16, 5, 3, 224, 224).
            x_out_ae (torch.Tensor): tensor containing reconstructed original images (batch, no_seq+1, c, h, w).
            x_out_lat (torch.Tensor): tensor containing predicted latent states (batch, seq, hidden_size).
            x_in_lat (torch.Tensor):  tensor containing latent representation of x_in.
        """

        xs = x_in.shape  # (batch, (seq+1), c, h, w)
        x_in = x_in.view(-1, xs[2], xs[3], xs[4])  # (batch*(seq+1),  c, h, w)

        # Encode all image sequence (n_batch x (n_seq+1)) -> latent representation
        x_in_lat = self.encode(x_in)  # (batch*(seq+1),  input_size)
        x_out_ae = self.decode(x_in_lat)  # Decode again
        x_in_lat = x_in_lat.view(xs[0], xs[1], -1)  # (batch, seq, input_size)

        # Initialize hidden and cell states for LSTM
        # (self.num_layers, batch_size, self.hidden_size)
        #h_0 = torch.zeros(self.num_layers, xs[0], self.hidden_size).to(self.device)
        # TODO: Number of layers
        if self.h_0_latent:
            h_0 = torch.unsqueeze(x_in_lat[:,0, :],0).contiguous()
        else:
            h_0 = torch.zeros(self.num_layers, xs[0], self.hidden_size).to(self.device)

        if self.rnn_arch == "lstm":
            c_0 = torch.zeros(self.num_layers, xs[0], self.hidden_size).to(self.device)
            # Predict based on (seq-1) frames
            x_out_lat, (h_0, c_0) = self.rnn(x_in_lat[:, :-1, :], (h_0, c_0))  # (batch, seq, hidden_size)
        elif self.rnn_arch == "gru":
            x_out_lat, h_0 = self.rnn(x_in_lat[:, :-1, :], h_0)  # (batch, seq, hidden_size)

        # TODO: Change input size to other when hidden size is different (add fc layers)/
        x_out = self.decode(x_out_lat.reshape(-1, self.hidden_size))
        x_out = x_out.view(xs[0], xs[1] - 1, xs[2], xs[3], xs[4])

        return x_out, x_out_ae, x_out_lat, x_in_lat


    def loss(self, out, target, n_batch):
        """Single function to calculate loss both for train and validation."""

        x_out, x_out_ae, x_out_lat, x_in_lat = out
        x_in = target

        xs = x_in.shape
        n = n_batch * self.no_seq

        # TODO: Optimize reshapes
        if self.loss_last_step:
            # Using last prediction only for loss
            x_in_temp  = x_in[:, -1, :, :, :].reshape(-1, xs[2], xs[3], xs[4])
            x_out_temp = x_out[:, -1, :, :, :].reshape(-1, xs[2], xs[3], xs[4])
        else:
            # Using all predicted frames for loss
            x_in_temp = x_in[:, 1:, :, :, :].reshape(-1, xs[2], xs[3], xs[4])
            x_out_temp = x_out.reshape(-1, xs[2], xs[3], xs[4])

        x_in_lat_temp = x_in_lat[:, 1:, :].reshape(-1, self.hidden_size)
        x_out_lat_temp = x_out_lat.reshape(-1, self.hidden_size)
        x_in_ae = x_in.reshape(-1, xs[2], xs[3], xs[4])  # Autoencoder input

        # Losses
        loss_image_1 = 1 - self.loss_msssim(x_out_ae, x_in_ae)
        loss_image_2 = 1 - self.loss_msssim(x_out_temp, x_in_temp)

        loss_latent = F.smooth_l1_loss(x_out_lat_temp, x_in_lat_temp, reduction='sum') / n
        loss = (loss_image_1 + loss_image_2) / 2.0 + loss_latent

        # No batch size division for MS-SSIM
        num_li = (loss_image_1.detach().cpu().numpy() + loss_image_2.detach().cpu().numpy()) / 2.0
        num_ll = loss_latent.detach().cpu().numpy()
        num_l = num_li + num_ll

        return loss, np.array([num_l, num_li, num_ll])

    def training_step(self, batch, batch_idx):

        x_out, x_out_ae, x_out_lat, x_in_lat = self.forward(batch[0])

        loss, loss_num = self.loss(
            (x_out, x_out_ae, x_out_lat, x_in_lat),
            batch[0],
            n_batch=batch[0].shape[0]
        )

        self.log('loss_train', loss_num[0], on_step=False, on_epoch=True, batch_size=batch[0].shape[0])
        self.log('loss_train_image', loss_num[1], on_step=False, on_epoch=True, batch_size=batch[0].shape[0])
        self.log('loss_train_latent', loss_num[2], on_step=False, on_epoch=True, batch_size=batch[0].shape[0])

        return loss


    def validation_step(self, batch, batch_idx):

        x_out, x_out_ae, x_out_lat, x_in_lat = self.forward(batch[0])

        _, loss_num = self.loss(
            (x_out, x_out_ae, x_out_lat, x_in_lat),
            batch[0],
            n_batch=batch[0].shape[0]
        )

        self.log('loss_val', loss_num[0], on_step=False, on_epoch=True, batch_size=batch[0].shape[0])
        self.log('loss_val_image', loss_num[1], on_step=False, on_epoch=True, batch_size=batch[0].shape[0])
        self.log('loss_val_latent', loss_num[2], on_step=False, on_epoch=True, batch_size=batch[0].shape[0])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

class SimpleLatentSensorPredictionNet(SimpleLatentPredictionNet):
    """ RNN: Latent + Sensor data
    """

    def __init__(self, config, autoencoder: torch.nn.Module, device):
        super().__init__(
            autoencoder,
            input_size=config["image_latent_size"] + len(config["sensor_fields"]),
            hidden_size=config["image_latent_size"] + len(config["sensor_fields"]),
            num_layers=config["lstm_n_layers"],
            dropout=config["dropout"],
            device=config["device"],
            rnn_arch=config["rnn_arch"]
        )

        self.device = device

        self.sensor_size = len(config["sensor_fields"])
        self.latent_size = config["image_latent_size"]

        # Sensor scaling layers
        self.fc_sensor_in = nn.Linear(in_features=self.sensor_size, out_features=self.sensor_size)
        self.fc_sensor_out = nn.Linear(in_features=self.sensor_size, out_features=self.sensor_size)

    def forward(self, x_in, s_in):
        """Forward function

        Args:
            x_in (torch.Tensor): input tensor of shape (batch_size, no_seq+1, C, H, W)
                Contains images at t_0 .. t_n, and target image at t_{n+1}.
                Example: (16, 6, 3, 224, 224), where sequence length is 5 and one image is predicted.
            s_in (torch.Tensor): input tensor of shape (batch_size, no_seq+1, M) where M is the number of sensor measurmeents.
        Returns:
            x_out (torch.Tensor): tensor containing predicted reconstructed images (batch, no_seq, c, h, w).
                Contains images at t_1 .. t_{n+1}. Example; (16, 5, 3, 224, 224).
            x_out_lat (torch.Tensor): tensor containing predicted latent states (batch, seq, hidden_size).
            x_in_lat (torch.Tensor):  tensor containing latent representation of x_in.
            s_out  (torch.Tensor): predicted sensor measurements (16, 5, M)
        """

        xs = x_in.shape  # (batch, (seq+1), c, h, w)
        ss = s_in.shape  # (batch, (seq+1), s)
        
        # Scale sensor data (input)
        s_in = s_in.view(-1, ss[2])     # (batch*(seq+1), s)
        s_in = self.fc_sensor_in(s_in)    
        s_in = s_in.view(ss[0], ss[1], ss[2])

        # Encode all image sequence (n_batch x (n_seq+1)) -> latent representation
        x_in = x_in.view(-1, xs[2], xs[3], xs[4])  # (batch*(seq+1),  c, h, w)
        x_in_lat = self.encode(x_in)  # (batch*(seq+1),  input_size)
        x_out_ae = self.decode(x_in_lat)  # Decode again
        x_in_lat = x_in_lat.view(xs[0], xs[1], -1)  # (batch, seq, input_size)
        
        # Concatenate latent representation with sensor measurements
        concat_in = torch.cat((x_in_lat, s_in), dim=2)  # (batch, seq, latent_size+sensor_size)

        if self.h_0_latent:
            h_0 = torch.unsqueeze(concat_in[:, 0, :],0).contiguous()
        else:
            h_0 = torch.zeros(self.num_layers, xs[0], self.latent_size + self.sensor_size).to(self.device)

        # Predict based on (seq-1) frames
        if self.rnn_arch == "lstm":
            c_0 = torch.zeros(self.num_layers, xs[0], self.latent_size + self.sensor_size).to(self.device)
            x_out_lat, (h_0, c_0) = self.rnn(x_in_lat[:, :-1, :], (h_0, c_0))  # (batch, seq, hidden_size)
        elif self.rnn_arch == "gru":
            concat_out, h_0 = self.rnn(concat_in[:, :-1, :], h_0)  # (batch, seq, hidden_size)

        # Scale sensor data (output)
        s_out = concat_out[:, :, -self.sensor_size :]  
        s_out = s_out.reshape(-1, ss[2])     # (batch*(seq+1), s)
        s_out = self.fc_sensor_out(s_out)    # Scale sensors out
        s_out = s_out.view(ss[0], ss[1]-1, ss[2])   # One less timestep, thus -1

        # TODO: Change input size to other when hidden size is different (add fc layers)/
        x_out_lat = concat_out[:, :, : -self.sensor_size]
        x_out = self.decode(x_out_lat.reshape(-1, self.latent_size))
        x_out = x_out.view(xs[0], xs[1] - 1, xs[2], xs[3], xs[4])

        return x_out, x_out_ae, x_out_lat, x_in_lat, s_out