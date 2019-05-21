import pickle
import os
from functools import partial

class AverageMeter:
    def __init__(self):
        self.value = None
        self.avg = None
        self.sum = None
        self.count = None
        self.all_values = None
        self.reset()

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.all_values = []

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count
        self.all_values.append(value)

class EarlyStopping:
    def __init__(self, mode='min', patience=20, threshold=1e-4, threshold_mode='rel'):
        self.mode = mode
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode

        self.num_bad_epochs = 0
        self.mode_worse = None  # the worse value for the chosen mode
        self.is_better = None
        self.last_epoch = -1
        self.is_converged = False
        self._init_is_better(mode=mode, threshold=threshold, threshold_mode=threshold_mode)
        self.best = self.mode_worse

    def step(self, metrics):
        if self.is_converged:
            raise ValueError
        current = metrics
        self.last_epoch += 1

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs > self.patience:
            self.is_converged = True

    def _cmp(self, mode, threshold_mode, threshold, a, best):
        if mode == 'min' and threshold_mode == 'rel':
            rel_epsilon = 1. - threshold
            return a < best * rel_epsilon
        elif mode == 'min' and threshold_mode == 'abs':
            return a < best - threshold
        elif mode == 'max' and threshold_mode == 'rel':
            rel_epsilon = threshold + 1.
            return a > best * rel_epsilon
        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = float('inf')
        else:  # mode == 'max':
            self.mode_worse = (-float('inf'))

        self.is_better = partial(self._cmp, mode, threshold_mode, threshold)


def get_model_dir(model_id):
    dumps_dir = '../dumps'
    return '{}/{}'.format(dumps_dir, model_id)

def get_pickle_file(model_dir, file_name_id):
    file_names = ['{}/{}'.format(model_dir, f) for f in os.listdir(model_dir) if file_name_id in f]
    
    if len(file_names) > 1:
        # Make sure we want training dumps
        assert '_test_' not in file_name_id and '_eval_' not in file_name_id
        file_names = [f for f in file_names if '_test_' not in f and '_eval_' not in f]
        if 'entropy' in file_name_id:
            if len(file_names) == 2: # language and non language entropies
                if len(file_names[0]) < len(file_names[1]):
                    file_names = [file_names[0]]
                else:
                    file_names = [file_names[1]]

    assert len(file_names) == 1

    file_name = file_names[0]

    return pickle.load(open(file_name, 'rb'))


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 30, 30)
    return x