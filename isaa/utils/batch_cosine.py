"""Single-cycle learning rates indexed by completed training updates."""
import math


class BatchWarmupCosine:
    def __init__(self, *, epochs, batches_per_epoch, warmup_epochs, base_lr,
                 warmup_start_lr, min_lr):
        self.total_updates = epochs * batches_per_epoch
        self.warmup_updates = warmup_epochs * batches_per_epoch
        if batches_per_epoch < 1 or not 0 <= warmup_epochs < epochs:
            raise ValueError("Cosine requires nonempty training batches and 0 <= warmup_epochs < epochs")
        if self.warmup_updates == 1 or self.total_updates - self.warmup_updates < 2:
            raise ValueError("Warmup needs zero or at least two updates; cosine needs at least two updates")
        if not all(math.isfinite(v) for v in (base_lr, warmup_start_lr, min_lr)):
            raise ValueError("Learning rates must be finite")
        if not 0 <= min_lr <= base_lr or not 0 < warmup_start_lr <= base_lr:
            raise ValueError("Require 0 <= min_lr <= base_lr and 0 < warmup_start_lr <= base_lr")
        self.batches_per_epoch = batches_per_epoch
        self.base_lr, self.warmup_start_lr, self.min_lr = base_lr, warmup_start_lr, min_lr
        self.completed_updates = 0

    def lr_at(self, update):
        if not 0 <= update < self.total_updates:
            raise ValueError("Training update is outside the configured cosine budget")
        if update < self.warmup_updates:
            return self.warmup_start_lr + (self.base_lr - self.warmup_start_lr) * update / (self.warmup_updates - 1)
        progress = (update - self.warmup_updates) / (self.total_updates - self.warmup_updates - 1)
        return self.min_lr + .5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

    def state_dict(self):
        return dict(name="warmup_cosine", version=1, step_unit="optimizer_update",
                    set_lr="before_training_batch", cosine_restarts=False, milestones=[],
                    batches_per_epoch=self.batches_per_epoch, total_updates=self.total_updates,
                    warmup_updates=self.warmup_updates, base_lr=self.base_lr,
                    warmup_start_lr=self.warmup_start_lr, min_lr=self.min_lr,
                    completed_updates=self.completed_updates)
