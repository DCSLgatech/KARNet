import torch
from torch import nn

def calculate_loss(a_in: torch.Tensor, a_out: torch.Tensor):
    """ Calculates cross-entropy loss. 
    
    Args: 
        a_in  (torch.Tensor): groundtruth actions
        a_out (torch.Tensor): output actions
    
    Returns:
        loss  (torch.nn.CrossEntropyLoss): criterion
        loss_num (np.array): criterion (numpy)
    """
    criterion = nn.CrossEntropyLoss()
    loss = criterion(a_out, a_in)
    loss_num = loss.detach().cpu().numpy()
    return loss, loss_num


class LatentToActionNet(nn.Module):
    """ Simple immitation learning net.
    """

    def __init__(self, latent_size: int, action_size: int):
        super(LatentToActionNet, self).__init__()

        self.latent_size = latent_size
        self.out_size = action_size

        #p=0.2
        # Latent to action space
        self.fc = nn.Sequential(
            nn.Linear(self.latent_size, int(self.latent_size/2.0)),
            #nn.Dropout(p=p),
            nn.ReLU(),
            nn.Linear(int(self.latent_size/2.0), int(self.latent_size/2.0)),
            #nn.Dropout(p=p),
            nn.ReLU(),
            nn.Linear(int(self.latent_size/2.0), int(self.latent_size/4.0)),
            #nn.Dropout(p=p),
            nn.ReLU(),
            nn.Linear(int(self.latent_size/4.0), 9),
        )


    def forward(self, x):
        x = self.fc(x)
        return x.squeeze()



class LatentToActionNetWithBackbone(nn.Module):
    """ Simple immitation learning net.
    """

    def __init__(self, backbone, latent_size: int, action_size: int):
        super(LatentToActionNetWithBackbone, self).__init__()

        self.latent_size = latent_size
        self.out_size = action_size

        self.backbone = backbone

        for param in backbone.parameters():
            param.requires_grad = False

        self.imitation = LatentToActionNet(latent_size, action_size)

    def forward(self, x):
        self.backbone.eval()
        x = self.backbone(x).squeeze()
        x = self.imitation(x)
        return x


class DynamicLatentToAction(nn.Module):
    """ Simple immitation learning net."""

    def __init__(self, backbone, latent_size: int, action_size: int):
        super(DynamicLatentToAction, self).__init__()

        self.latent_size = latent_size
        self.action_size = action_size

        self.backbone = backbone

        for param in backbone.parameters():
            param.requires_grad = False

        self.imitation = LatentToActionNet(latent_size, action_size)


    def forward(self, x):
        
        # x_out         (n_batch, seq, channels, size_x, size_y)
        # x_out_lat     (n_batch, seq, latent_size)
        # x_in_lat      (n_batch, seq+1, latent_size)
        self.backbone.eval()
        
        _, _, x_out_lat, _ = self.backbone(x)
        x = self.imitation(x_out_lat[:, -1, :])
        return x.squeeze()


class DynamicLatentToActionTwo(nn.Module):
    """ Simple immitation learning net."""

    def __init__(self, backbone, latent_size: int, action_size: int):
        super(DynamicLatentToActionTwo, self).__init__()

        self.latent_size = latent_size * 2
        self.action_size = action_size

        self.backbone = backbone

        for param in backbone.parameters():
            param.requires_grad = False

        # self.imitation_c1d = nn.Sequential(
        #     nn.Conv1d(in_channels=2, out_channels=1, kernel_size=3, padding=1),
        #     #nn.LeakyReLU(negative_slope=0.1)
        # )

        self.imitation = LatentToActionNet(self.latent_size, self.action_size)

    def forward(self, x):
        
        # x_out         (n_batch, seq, channels, size_x, size_y)
        # x_out_lat     (n_batch, seq, latent_size)
        # x_in_lat      (n_batch, seq+1, latent_size)
        self.backbone.eval()
        
        # x_out, x_out_ae, x_out_lat, x_in_lat
        _, _, x_out_lat, _ = self.backbone(x)
        # TODO: Check immitation

        x = torch.cat((x_out_lat[:,-2,:], x_out_lat[:,-1,:]), dim=1)

        #x = self.imitation_c1d(x_out_lat[:, -2:, :])
        x = self.imitation(x)
        return x.squeeze()