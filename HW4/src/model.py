import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, in_dim, num_freqs, include_input=True):
        super().__init__()
        self.in_dim = in_dim
        self.num_freqs = num_freqs
        self.include_input = include_input

        self.out_dim = 0
        if include_input:
            self.out_dim += in_dim
        self.out_dim += 2 * in_dim * num_freqs

    def forward(self, x):
        outputs = []
        if self.include_input:
            outputs.append(x)

        for k in range(self.num_freqs):
            freq = (2.0**k) * math.pi
            outputs.append(torch.sin(freq * x))
            outputs.append(torch.cos(freq * x))

        return torch.cat(outputs, dim=-1)


class NeRF(nn.Module):
    def __init__(
        self,
        L_xyz=10,
        L_dir=4,
        hidden_dim=256,
    ):
        super().__init__()

        self.xyz_pe = PositionalEncoding(in_dim=3, num_freqs=L_xyz, include_input=True)
        self.dir_pe = PositionalEncoding(in_dim=3, num_freqs=L_dir, include_input=True)

        xyz_dim = self.xyz_pe.out_dim
        dir_dim = self.dir_pe.out_dim

        # Shared trunk
        self.fc1 = nn.Linear(xyz_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)

        # Skip connection after 4 layers
        self.fc5 = nn.Linear(hidden_dim + xyz_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, hidden_dim)
        self.fc7 = nn.Linear(hidden_dim, hidden_dim)
        self.fc8 = nn.Linear(hidden_dim, hidden_dim)

        # Density head
        self.sigma_fc = nn.Linear(hidden_dim, 1)

        # RGB head
        self.feature_fc = nn.Linear(hidden_dim, hidden_dim)
        self.rgb_fc1 = nn.Linear(hidden_dim + dir_dim, 128)
        self.rgb_fc2 = nn.Linear(128, 3)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, xyz, ray_d):
        """
        Args:
            xyz:   (..., 3) sample positions in world coordinates
            ray_d: either (..., 3) or one dimension fewer
                   e.g. xyz=(N,S,3), ray_d=(N,3)

        Returns:
            rgbs:   (..., 3)
            sigmas: (..., 1)
        """
        original_shape = xyz.shape[:-1]

        if ray_d.dim() == xyz.dim() - 1:
            ray_d = ray_d.unsqueeze(-2).expand(*xyz.shape[:-1], 3)

        xyz_flat = xyz.reshape(-1, 3)
        ray_d_flat = ray_d.reshape(-1, 3)

        xyz_encoded = self.xyz_pe(xyz_flat)
        dir_encoded = self.dir_pe(ray_d_flat)

        h = self.relu(self.fc1(xyz_encoded))
        h = self.relu(self.fc2(h))
        h = self.relu(self.fc3(h))
        h = self.relu(self.fc4(h))

        h = torch.cat([h, xyz_encoded], dim=-1)

        h = self.relu(self.fc5(h))
        h = self.relu(self.fc6(h))
        h = self.relu(self.fc7(h))
        h = self.relu(self.fc8(h))

        sigmas = torch.nn.functional.softplus(self.sigma_fc(h))

        features = self.relu(self.feature_fc(h))
        rgb_input = torch.cat([features, dir_encoded], dim=-1)
        rgb_hidden = self.relu(self.rgb_fc1(rgb_input))
        rgbs = self.sigmoid(self.rgb_fc2(rgb_hidden))

        rgbs = rgbs.reshape(*original_shape, 3)
        sigmas = sigmas.reshape(*original_shape, 1)

        return rgbs, sigmas
