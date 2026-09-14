"""Reference rgb-depth-model-training solution: train a small CNN.

Runs in the agent container (GPU). Dataset generation and epoch-controlled
optimization use the shipped harness API.
"""
import json
import sys

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, "/workspace")

from harness import spec, torch_contract, train_harness
sys.path.insert(0, "/solution/payload")
from rbp.reference_labels import label_shape


class PoseNet(nn.Module):
    """Calibrated object-centric pose network."""

    def __init__(self):
        super().__init__()
        self.crop_size = 160
        self.crop_half_pixels = 112.0
        def block(cin, cout, stride):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True))
        self.net = nn.Sequential(
            block(6, 32, 2), block(32, 64, 2), block(64, 96, 2),
            block(96, 128, 2), block(128, 160, 2),
            nn.AdaptiveAvgPool2d(1))
        self.head = nn.Sequential(
            nn.Linear(166, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 4))
        self.shape_head = nn.Linear(166, 3)
        self.fx = float(spec.intrinsics()[0, 0])
        self.fy = float(spec.intrinsics()[1, 1])
        self.cx = float(spec.intrinsics()[0, 2])
        self.cy = float(spec.intrinsics()[1, 2])
        self.register_buffer(
            "camera_rotation",
            torch.as_tensor(spec.camera_rotation(), dtype=torch.float32))
        self.register_buffer(
            "camera_position",
            torch.as_tensor(spec.CAM_POS, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        red, green, blue = x[:, 0], x[:, 1], x[:, 2]
        mask = ((red > (60.0 / 255.0))
                & (red > 1.35 * green)
                & (red > 1.35 * blue)).to(dtype=x.dtype)

        rows = torch.arange(height, dtype=x.dtype, device=x.device)
        cols = torch.arange(width, dtype=x.dtype, device=x.device)
        mass = mask.sum(dim=(1, 2)).clamp_min(1.0)
        u = (mask.sum(dim=1) * cols.unsqueeze(0)).sum(dim=1) / mass
        v = (mask.sum(dim=2) * rows.unsqueeze(0)).sum(dim=1) / mass

        du = (cols.unsqueeze(0) - u.unsqueeze(1)) / self.crop_half_pixels
        dv = (rows.unsqueeze(0) - v.unsqueeze(1)) / self.crop_half_pixels
        col_mass = mask.sum(dim=1)
        row_mass = mask.sum(dim=2)
        var_u = (col_mass * du.square()).sum(dim=1) / mass
        var_v = (row_mass * dv.square()).sum(dim=1) / mass
        cov_uv = (
            mask * dv.unsqueeze(2) * du.unsqueeze(1)
        ).sum(dim=(1, 2)) / mass

        axis = torch.linspace(-1.0, 1.0, self.crop_size,
                              dtype=x.dtype, device=x.device)
        grid_u = (u[:, None, None]
                  + self.crop_half_pixels * axis[None, None, :])
        grid_v = (v[:, None, None]
                  + self.crop_half_pixels * axis[None, :, None])
        grid_u = grid_u.expand(batch, self.crop_size, self.crop_size)
        grid_v = grid_v.expand(batch, self.crop_size, self.crop_size)
        grid = torch.stack((2.0 * grid_u / float(width - 1) - 1.0,
                            2.0 * grid_v / float(height - 1) - 1.0), dim=-1)
        crop_input = torch.cat((x, mask.unsqueeze(1)), dim=1)
        crop = F.grid_sample(crop_input, grid, mode="bilinear",
                             padding_mode="zeros", align_corners=True)
        visual = self.net(crop).flatten(1)

        d_camera = torch.stack(((u - self.cx) / self.fx,
                                -(v - self.cy) / self.fy,
                                -torch.ones_like(u)), dim=1)
        d_world = d_camera @ self.camera_rotation.t()
        scale = ((spec.BLOCK_HEIGHT / 2.0 - self.camera_position[2])
                 / d_world[:, 2].clamp_max(-1e-6))
        center_xy = (self.camera_position[:2].unsqueeze(0)
                     + scale.unsqueeze(1) * d_world[:, :2])

        geometry = torch.stack((u / float(width) - 0.5,
                                v / float(height) - 0.5,
                                mass / float(height * width) * 50.0,
                                var_u, cov_uv, var_v), dim=1)
        features = torch.cat((visual, geometry), dim=1)
        raw = self.head(features)
        xy = center_xy + 0.15 * torch.tanh(raw[:, :2])
        return torch.cat((xy, raw[:, 2:4], self.shape_head(features)), dim=1)

def main():
    out = sys.argv[1]
    data_dir = "/tmp/dataset"
    print("generating dataset ...")
    train_harness.generate_dataset(data_dir, seeds=spec.DESIGN_SEEDS,
                                   frames_per_seed=300, episode_frames=100)
    torch.manual_seed(0)
    model = PoseNet()
    print("training for 54 epochs ...")
    report = train_harness.fit(model, data_dir, epochs=54,
                               batch_size=32, lr=1e-3, seed=0,
                               report_path=out + "/training_report.json",
                               shape_labeler=label_shape)
    print(json.dumps(report, indent=2))
    torch_contract.save_model(model, out + "/model.pt")

    print("payload staged")


if __name__ == "__main__":
    main()
