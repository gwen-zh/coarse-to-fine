"""EMA residual quantization and a capacity-matched parallel-prefix control."""
import torch
from torch import nn
from torch.nn import functional as F


class PrefixQuantizer(nn.Module):
    def __init__(self, sizes=(32, 512, 4096), width=128, mode='residual', decay=0.99):
        super().__init__()
        if mode not in ('residual', 'parallel'):
            raise ValueError(mode)
        self.sizes, self.width, self.mode, self.decay = tuple(sizes), width, mode, decay
        for level, size in enumerate(sizes):
            self.register_buffer(f'centers_{level}', torch.zeros(size, width))
            self.register_buffer(f'counts_{level}', torch.ones(size))
            self.register_buffer(f'sums_{level}', torch.zeros(size, width))

    @staticmethod
    def nearest(z, centers):
        distance = z.square().sum(-1, keepdim=True) - 2 * z @ centers.T + centers.square().sum(-1)
        return distance.argmin(-1)

    @torch.no_grad()
    def initialize(self, z, seed=0):
        generator = torch.Generator(device=z.device).manual_seed(seed)
        residual = z.clone()
        for level, size in enumerate(self.sizes):
            source = residual if self.mode == 'residual' else z
            if len(z) >= size:
                choice = torch.randperm(len(z), generator=generator, device=z.device)[:size]
            else:
                choice = torch.randint(len(z), (size,), generator=generator, device=z.device)
            centers = getattr(self, f'centers_{level}')
            centers.copy_(source[choice])
            getattr(self, f'sums_{level}').copy_(centers)
            getattr(self, f'counts_{level}').fill_(1)
            if self.mode == 'residual':
                for start in range(0, len(z), 1024):
                    part = source[start:start + 1024]
                    ids = self.nearest(part, centers)
                    residual[start:start + 1024] -= centers[ids]

    @torch.no_grad()
    def forward(self, z, update=False):
        prefixes, tokens = [], []
        accumulated = torch.zeros_like(z)
        for level, size in enumerate(self.sizes):
            source = z - accumulated if self.mode == 'residual' else z
            centers = getattr(self, f'centers_{level}')
            indices = self.nearest(source, centers)
            values = centers[indices]
            accumulated = accumulated + values
            prefix = accumulated if self.mode == 'residual' else accumulated / (level + 1)
            prefixes.append(prefix.clone())
            tokens.append(indices)
            if update:
                counts = torch.bincount(indices, minlength=size).to(z.dtype)
                sums = torch.zeros_like(centers).index_add_(0, indices, source)
                ema_counts = getattr(self, f'counts_{level}')
                ema_sums = getattr(self, f'sums_{level}')
                ema_counts.mul_(self.decay).add_(counts, alpha=1 - self.decay)
                ema_sums.mul_(self.decay).add_(sums, alpha=1 - self.decay)
                centers.copy_(ema_sums / ema_counts.clamp_min(1e-5)[:, None])
        return prefixes, tokens

    @torch.no_grad()
    def restart_unused(self, z, seed):
        """Replace near-empty EMA entries from training data/residuals only."""
        generator = torch.Generator(device=z.device).manual_seed(seed)
        accumulated = torch.zeros_like(z)
        restarts = []
        for level, size in enumerate(self.sizes):
            source = z - accumulated if self.mode == 'residual' else z
            centers = getattr(self, f'centers_{level}')
            counts = getattr(self, f'counts_{level}')
            unused = counts < 0.1
            number = int(unused.sum())
            choice = torch.randint(len(z), (number,), generator=generator, device=z.device)
            centers[unused] = source[choice]
            counts[unused] = 1
            getattr(self, f'sums_{level}')[unused] = centers[unused]
            for start in range(0, len(z), 1024):
                ids = self.nearest(source[start:start + 1024], centers)
                accumulated[start:start + 1024] += centers[ids]
            restarts.append(number)
        return restarts


class DecoderAdapter(nn.Module):
    def __init__(self, width=128, hidden=512, output=1280):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, output))

    def forward(self, prefix):
        return self.net(prefix)
