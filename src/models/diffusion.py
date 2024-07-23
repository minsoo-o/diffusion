import torch
import torch.nn as nn
import torch.nn.functional as F

from models.attentions import SelfAttention, CrossAttention
from models.blocks import ResidualBlock, AttentionBlock
from models.utils import TimeEmbedding
from training.config import TrainConfig


class SwitchSequential(nn.Sequential):
    def forward(self, x, t, y):
        for layer in self:
            if isinstance(layer, AttentionBlock):
                x = layer(x, y)
            elif isinstance(layer, ResidualBlock):
                x = layer(x, t)
            else:
                x = layer(x)
        return x
    
    
class UNet(nn.Module):
    def __init__(self, channels=[32, 64, 128, 256], n_attn_head=8, t_emb_dim=256, y_emb_dim=256) -> None:
        super().__init__()
        self.t_emb_dim = t_emb_dim
        self.in_conv = nn.Conv2d(1, channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.encoder_layers = nn.ModuleList(
            [
                SwitchSequential(
                    ResidualBlock(channels[i], channels[i+1], t_emb_dim=t_emb_dim),
                    AttentionBlock(n_attn_head, channels[i+1]//n_attn_head, y_emb_dim=y_emb_dim),
                    nn.Conv2d(channels[i+1], channels[i+1], kernel_size=3, stride=2, padding=1)
                ) for i in range(len(channels)-1)
            ] + [
                SwitchSequential(
                    ResidualBlock(channels[-1], channels[-1], t_emb_dim=t_emb_dim))   
            ] + [
                SwitchSequential(
                    ResidualBlock(channels[-1], channels[-1], t_emb_dim=t_emb_dim))
            ])        
        self.decoder_layers = nn.ModuleList(
            [
                SwitchSequential(
                    ResidualBlock(channels[-1]*2, channels[-1], t_emb_dim=t_emb_dim))
            ] + [
                SwitchSequential(
                    ResidualBlock(channels[-1]*2, channels[-1], t_emb_dim=t_emb_dim))
            ] + [
                SwitchSequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                    ResidualBlock(channels[i]*2, channels[i-1], t_emb_dim=t_emb_dim),
                    AttentionBlock(n_attn_head, channels[i-1]//n_attn_head, y_emb_dim=y_emb_dim),
                ) for i in range(len(channels)-1, 0, -1)
            ])

    def forward(self, x, t_embed, y_embed):
        x = self.in_conv(x)

        # Encoding
        skip_connections = []
        for i, layer in enumerate(self.encoder_layers):
            x = layer(x, t_embed, y_embed)
            skip_connections.append(x)

        # Decoding
        for layer in self.decoder_layers:
            skip_connection = skip_connections.pop()
            if skip_connection.size() != x.size():
                x = F.interpolate(x, size=skip_connection.size()[2:], mode='bilinear', align_corners=False)
            x = torch.cat((x, skip_connection), dim=1)
            x = layer(x, t_embed, y_embed)
        return x       


class Diffusion(nn.Module):
    def __init__(self, config: TrainConfig):
        super().__init__()
        self.config = config
        n_classes = config.n_classes
        in_channels = config.in_channels
        channels = config.channels
        n_attn_head = config.n_attn_head
        t_emb_dim = config.t_emb_dim
        y_emb_dim = config.y_emb_dim

        self.t_emb_layer = nn.Sequential(
            TimeEmbedding(t_emb_dim),
            nn.Linear(t_emb_dim, t_emb_dim),
            nn.GELU()
        )
        self.y_emb_layer = nn.Embedding(n_classes, y_emb_dim)
        self.unet = UNet(channels=channels, n_attn_head=n_attn_head, t_emb_dim=t_emb_dim, y_emb_dim=y_emb_dim)
        self.final = nn.Sequential(
            nn.GroupNorm(1, channels[0]),
            nn.GELU(),
            nn.Conv2d(channels[0], in_channels, kernel_size=3, padding=1),
        )
    
    def forward(self, x, t, y):
        t_embed = self.t_emb_layer(t)
        y_embed = self.y_emb_layer(y).unsqueeze(1)
        output = self.unet(x, t_embed, y_embed)
        output = self.final(output)
        return output


if __name__ == "__main__":    
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    n_attn_head = 8
    batch_size = 4
    in_channels = 1
    n_classes = 10

    model = Diffusion(n_classes=n_classes, n_attn_head=n_attn_head, in_channels=in_channels)
    input_values = torch.rand((batch_size, in_channels, 28, 28))
    labels = torch.randint(0, n_classes, (batch_size,), dtype=torch.int64)
    output = model(input_values, torch.tensor((0,)), labels)
    print(output)

