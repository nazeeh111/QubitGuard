"""Original compact axial spatiotemporal logical-frame network."""
import torch
from torch import nn

class FrameNet(nn.Module):
    def __init__(self, distance, width=12, variant='residual'):
        super().__init__()
        self.distance=distance; self.width=width; self.variant=variant
        temporal=1 if variant=='no_time' else 3
        # Factor spatial and temporal mixing; keep readout boundary-aware.
        self.encoder=nn.Sequential(
            nn.Conv3d(2,width,(1,3,3),padding=(0,1,1)),nn.SiLU(),
            nn.Conv3d(width,width,(temporal,1,1),padding=(temporal//2,0,0),groups=width),nn.SiLU(),
            nn.Conv3d(width,width,(1,3,3),padding=(0,1,1)),nn.SiLU(),
            nn.Conv3d(width,width,(temporal,1,1),padding=(temporal//2,0,0),groups=width),nn.SiLU())
        # Temporal statistics discard order for no_time; identical readout shape.
        self.pool=nn.AdaptiveAvgPool3d((2,2,2))
        self.head=nn.Sequential(nn.Linear(width*8+5,48),nn.SiLU(),nn.Linear(48,1))
    def forward(self, grid, context):
        if self.variant=='no_time':
            # Strict time-order ablation: repeat mean input over time.
            grid=grid.mean(dim=2,keepdim=True).expand_as(grid)
        if self.variant=='no_correlation':
            context=context.clone(); context[:,1:3]=0
        if self.variant=='direct':
            # Direct decoder receives no matching predictions.
            context=context.clone(); context[:,:3]=0
        h=self.pool(self.encoder(grid)).flatten(1)
        return self.head(torch.cat([h,context],dim=1)).squeeze(1)
    @property
    def parameters_count(self):
        return sum(p.numel() for p in self.parameters())
