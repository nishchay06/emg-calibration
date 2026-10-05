"""Pinned Torch-compatible scheduler whose clock is optimizer updates."""
from torch.optim.lr_scheduler import _LRScheduler

from calibration_protocol import learning_rate_at_update, validate_profile


class UpdateWarmupCosine(_LRScheduler):
    def __init__(self, optimizer, profile, last_epoch=-1):
        self.profile = dict(profile)
        validate_profile(self.profile)
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        return [learning_rate_at_update(self.profile, self.last_epoch)
                for _ in self.optimizer.param_groups]
