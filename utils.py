from torch.optim.lr_scheduler import ReduceLROnPlateau
from functools import partial

class AverageMeter:
    def __init__(self):
        self.value = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count


def get_lr_scheduler(optimizer):
    def reduce_lr(self, epoch):
        ReduceLROnPlateau._reduce_lr(self, epoch)

    lr_scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.2, patience=10, threshold=0.005,
                                     threshold_mode="rel")
    lr_scheduler._reduce_lr = partial(reduce_lr, lr_scheduler)
    return lr_scheduler