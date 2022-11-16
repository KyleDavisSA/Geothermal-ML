import torch
import torch.nn as nn

class SVDEncoder(nn.Module):
    def __init__(self, U_matrix) -> None:
        super().__init__()

        self.U_matrix = U_matrix

    def forward(self, x):
        # project onto basis
        return torch.matmul(x, self.U_matrix)


class SVDDecoder(nn.Module):
    def __init__(self, U_matrix) -> None:
        super().__init__()

        self.U_matrix = U_matrix

    def forward(self, x):
        # unproject input
        return torch.matmul(x, self.U_matrix.T)


class GWHPSVDModel(nn.Module):
    def __init__(self, U_tensor, num_modes: int) -> None:
        super().__init__()

        self.vel_x_e = SVDEncoder(U_tensor[0, :, :num_modes])
        self.vel_y_e = SVDEncoder(U_tensor[1, :, :num_modes])
        # self.temp_e = SVDEncoder(U_tensor[2, :, :num_modes])
        self.perm_e = SVDEncoder(U_tensor[3, :, :num_modes])
        self.pres_e = SVDEncoder(U_tensor[4, :, :num_modes])

        self.lin1 = nn.Linear(num_modes * 2, 256)
        self.lin2 = nn.Linear(256, 1024)
        self.lin3 = nn.Linear(1024, 4096)

        self.relu1 = nn.Tanh()
        self.relu2 = nn.Tanh()
        # self.relu3 = nn.ReLU()

    def forward(self, x):
        # expect x to contain vel, perm and press

        vel_x_r = self.vel_x_e(x[:, 0, :])
        vel_y_r = self.vel_y_e(x[:, 1, :])
        # perm_r = self.perm_e(x[:, 3, :])
        # pres_r = self.pres_e(x[:, 4, :])

        # encoded = torch.concat([vel_x_r, vel_y_r, perm_r, pres_r], dim=1)
        encoded = torch.concat([vel_x_r, vel_y_r], dim=1)

        linear1 = self.lin1(encoded)
        act1 = self.relu1(linear1)
        linear2 = self.lin2(act1)
        act2 = self.relu2(linear2)
        linear3 = self.lin3(act2)

        return linear3


class GWHPSVDEncodeDecode(nn.Module):
    def __init__(self, U_tensor, n_modes: int, n_hidden: int, n_latent_size: int) -> None:
        super().__init__()

        self.vel_x_e = SVDEncoder(U_tensor[0, :, :n_modes])
        self.vel_y_e = SVDEncoder(U_tensor[1, :, :n_modes])
        self.perm_e = SVDEncoder(U_tensor[3, :, :n_modes])
        self.pres_e = SVDEncoder(U_tensor[4, :, :n_modes])

        self.temp_d = SVDDecoder(U_tensor[2, :, :n_modes])

        self.hidden_layers = []
        for i in range(n_hidden):
            self.hidden_layers.append(nn.Linear(n_latent_size, n_latent_size))
            self.hidden_layers.append(nn.ReLU())

        self.fc = nn.Sequential(
            nn.Linear(n_modes * 4, 64),
            nn.Linear(64, n_latent_size),
            nn.ReLU(),
            *self.hidden_layers,
            nn.Linear(n_latent_size, 64),
            nn.ReLU(),
            nn.Linear(64, n_modes),
        )

    def forward(self, x):
        # expect x to contain vel, perm and press

        vel_x_r = self.vel_x_e(x[:, 0, :])
        vel_y_r = self.vel_y_e(x[:, 1, :])
        perm_r = self.perm_e(x[:, 3, :])
        pres_r = self.pres_e(x[:, 4, :])

        encoded = torch.concat([vel_x_r, vel_y_r, perm_r, pres_r], dim=1)
        # encoded = torch.concat([vel_x_r, vel_y_r], dim=1)

        latent = self.fc(encoded)

        decoded = self.temp_d(latent)

        return decoded


class GWHPSVDEncodeDecodeLinear(nn.Module):
    def __init__(self, U_tensor, num_modes: int) -> None:
        super().__init__()

        self.vel_x_e = SVDEncoder(U_tensor[0, :, :num_modes])
        self.vel_y_e = SVDEncoder(U_tensor[1, :, :num_modes])
        self.perm_e = SVDEncoder(U_tensor[3, :, :num_modes])
        self.pres_e = SVDEncoder(U_tensor[4, :, :num_modes])

        self.temp_d = SVDDecoder(U_tensor[2, :, :num_modes])

        self.fc = nn.Sequential(nn.Linear(num_modes * 2, num_modes, bias=True))

    def forward(self, x):
        # expect x to contain vel, perm and press

        vel_x_r = self.vel_x_e(x[:, 0, :])
        vel_y_r = self.vel_y_e(x[:, 1, :])
        # perm_r = self.perm_e(x[:, 3, :])
        # pres_r = self.pres_e(x[:, 4, :])

        # encoded = torch.concat([vel_x_r, vel_y_r, perm_r, pres_r], dim=1)
        encoded = torch.concat([vel_x_r, vel_y_r], dim=1)

        latent = self.fc(encoded)

        decoded = self.temp_d(latent)

        return decoded
