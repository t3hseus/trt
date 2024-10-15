"""
Used code from:
https://github.com/yanx27/Pointnet_Pointnet2_pytorch
"""
import torch.nn as nn
import torch.nn.functional as F
import torch
from src.deprecated.pointnet.pointnet_utils import (
    PointNetSetAbstraction,
    PointNetFeaturePropagation,
)


class PointNet2(nn.Module):
    def __init__(
        self, num_points: int = 1024, base_radius: float = 0.1, in_channel: int = 3
    ):
        super(PointNet2, self).__init__()
        self.sa1 = PointNetSetAbstraction(
            num_points, base_radius, 32, 3 + in_channel, [32, 32, 64], False
        )
        self.sa2 = PointNetSetAbstraction(
            num_points // 4, base_radius * 2, 32, 64 + in_channel, [64, 64, 128], False
        )
        self.sa3 = PointNetSetAbstraction(
            num_points // 16,
            base_radius * 4,
            32,
            128 + in_channel,
            [128, 128, 256],
            False,
        )
        self.sa4 = PointNetSetAbstraction(
            num_points // 64,
            base_radius * 8,
            32,
            256 + in_channel,
            [256, 256, 512],
            False,
        )
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)

    def forward(self, inputs: torch.Tensor):
        """
        Input:
            xyz: input points position data, [B, C, N]
        Return:
            new_xyz: result points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        l0_points = inputs
        coords = inputs[:, :3, :]

        l1_xyz, l1_points = self.sa1(coords, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(coords, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        return x, l4_points


if __name__ == "__main__":
    model = PointNet2()
    xyz = torch.rand(10, 3, 2048)
    res = model(xyz)
    print(res[0].shape, res[1].shape)
