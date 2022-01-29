import torch
import warnings
from torch_geometric.data import Data, Batch
import copy

class ExponentialLR_with_minLr(torch.optim.lr_scheduler.ExponentialLR):
    def __init__(self, optimizer, gamma, min_lr=1e-4, last_epoch=-1, verbose=False):
        self.gamma = gamma
        self.min_lr = min_lr
        super(ExponentialLR_with_minLr, self).__init__(optimizer, gamma, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0:
            return self.base_lrs
        return [max(group['lr'] * self.gamma, self.min_lr)
                for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [max(base_lr * self.gamma ** self.last_epoch, self.min_lr)
                for base_lr in self.base_lrs]



def clip_norm(vec, limit, p=2):
    norm = torch.norm(vec, dim=-1, p=2, keepdim=True)
    denom = torch.where(norm > limit, limit / norm, torch.ones_like(norm))
    return vec * denom

def repeat_data(data: Data, num_repeat) -> Batch:
    datas = [copy.deepcopy(data) for i in range(num_repeat)]
    return Batch.from_data_list(datas)


def repeat_batch(batch: Batch, num_repeat) -> Batch:
    datas = batch.to_data_list()
    new_data = []
    for i in range(num_repeat):
        new_data += copy.deepcopy(datas)
    return Batch.from_data_list(new_data)



#@title get_scheduler
def get_scheduler(config, optimizer):
    if config.train.scheduler.type == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=config.train.scheduler.factor,
            patience=config.train.scheduler.patience,
        )
    elif config.train.scheduler == 'expmin':
        return ExponentialLR_with_minLr(
            optimizer,
            gamma=config.factor,
            min_lr=config.min_lr,
        )
    else:
        raise NotImplementedError('Scheduler not supported: %s' % config.type)