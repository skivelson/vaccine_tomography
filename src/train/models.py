import torch
from torch import nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics import Metric


import sys


class DiceMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("intersection", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("denom", default=torch.tensor(1e-7), dist_reduce_fx="sum")

    def update(self, y_pred, y_true):
        y_pred = torch.sigmoid(y_pred)
        y_pred = torch.where(y_pred < 0.9, 0.0, 1.0)

        self.intersection += torch.sum(y_true * y_pred)
        self.denom += torch.sum(y_true) + torch.sum(y_pred)

    def compute(self):
        return 2 * self.intersection / self.denom


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        intersection = torch.sum(y_true * y_pred)
        denom = torch.sum(y_true) + torch.sum(y_pred)
        dice_loss = 1 - (2 * intersection + 1e-7) / (denom + 1e-7)

        return dice_loss


class AnalysisBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv3d(in_channels, out_channels, 3, padding="same"),
            nn.ELU(),
            nn.Conv3d(out_channels, out_channels, 3, padding="same"),
            nn.ELU(),
            nn.GroupNorm(8, out_channels),
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class SynthesisBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose3d(in_channels, out_channels, 2, stride=2)
        self.elu = nn.ELU()
        
        self.layers = nn.ModuleList([
            nn.Conv3d(in_channels, out_channels, 3, padding="same"),
            nn.ELU(),
            nn.Conv3d(out_channels, out_channels, 3, padding="same"),
            nn.ELU(),
            nn.GroupNorm(8, out_channels),
        ])
        
    def forward(self, inputs):
        x, skip_x = inputs
        x = self.elu(self.upconv(x))
        x = torch.cat([x, skip_x], 1)  # channel concat

        for layer in self.layers:
            x = layer(x)
        return x
    

class UNet3D(LightningModule):
    def __init__(self, channels=[8, 16, 32, 64, 128, 256]):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        self.metric = DiceMetric()
        
        self.pool = nn.MaxPool3d(2)
        self.bottom_layer = AnalysisBlock(channels[-2], channels[-1])
        self.output_layer = nn.Conv3d(channels[0], 1, 1, padding="same")

        analysis_layers = []
        synthesis_layers = []
        channels = [1] + channels
        for i in range(len(channels) - 2):
            analysis_layers.append(AnalysisBlock(channels[i], channels[i+1]))
            synthesis_layers.append(SynthesisBlock(channels[i+2], channels[i+1]))
        synthesis_layers.reverse()
       
        self.analysis_layers = nn.ModuleList(analysis_layers)
        self.synthesis_layers = nn.ModuleList(synthesis_layers)

    def forward(self, x):
        analysis_outputs = []

        for layer in self.analysis_layers:
            x = layer(x)
            analysis_outputs.append(x)
            x = self.pool(x)

        x = self.bottom_layer(x)

        for layer, skip_x in zip(self.synthesis_layers, analysis_outputs[::-1]):
            x = layer([x, skip_x])

        return self.output_layer(x)

    def training_step(self, batch, batch_idx):
        x = batch["data"]
        y_true = batch["labels"]
        weight = batch["weight"]
        
        weight = weight.view(-1, 1).detach()
        y_pred = self(x)

        y_pred = torch.masked_select(y_pred.view(-1, 1), weight)
        y_true = torch.masked_select(y_true.view(-1, 1), weight)

        dice_loss = self.dice_loss(torch.sigmoid(y_pred), y_true)
        bce_loss = self.bce_loss(y_pred, y_true)
        loss = dice_loss + bce_loss

        self.log("dice_loss", dice_loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("bce_loss", bce_loss, prog_bar=True, on_epoch=True, on_step=False)

        return {
            "loss": loss,
            "y_pred": y_pred.detach(),
            "y_true": y_true,
        }

    def training_step_end(self, outputs):
        self.metric(outputs["y_pred"], outputs["y_true"])
        self.log("dice_score", self.metric, prog_bar=True, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)



if __name__ == '__main__':
    pass
