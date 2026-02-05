from torch import nn

from .graph_modules import (
    GraphEncoder,
    GraphDecoder,
    GraphDenoiserMasked,
)

class Diffusion1D(nn.Module):
    def __init__(self):
        super().__init__()

        # Graph Encoder/Decoder
        self.graph_encoder = GraphEncoder(d_model=128, heads=4, depth=1, dropout=0.1, hops=1)
        self.graph_decoder = GraphDecoder(d_model=128, heads=4, depth=1, dropout=0.1, hops=1)

        # Denoiser
        self.unet = GraphDenoiserMasked(time_emb_dim=12, context_dim=512, num_classes=2, hops=1)

        self.final = nn.Conv1d(90, 90, kernel_size=1, padding=0)

        self.apply(self.initialize_weights)

    def initialize_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, latent, context, time, sensor_pred):
        latent = latent.to(context.device)
        time = time.to(context.device)
        sensor_pred = sensor_pred.to(context.device)

        z0 = self.graph_encoder(latent)
        zt = self.unet(z0, context, time, sensor_pred)
        xhat = self.graph_decoder(zt)

        return self.final(xhat)
