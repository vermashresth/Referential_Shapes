import numpy as np
import torch
import random
from torch.utils.data.sampler import Sampler
import torchvision.transforms
from PIL import Image

class ImageDataset():
    def __init__(self, file_name, mean=None, std=None):
        print("Image Dataset loading file")
        self.pixels = np.load(file_name)
        data_len = len(self.pixels)
        sample_ind = np.random.choice(data_len, 1000, replace=false)
        self.sample = self.pixels[sample_ind]
        print("Loaded npy file")

        self.use_different_targets = self.pixels.shape[1] == 2
        print("Calculating mean")

        if mean is None:
            mean = np.mean(self.sample, axis=tuple(range(self.sample.ndim-1)))
            std = np.std(self.sample, axis=tuple(range(self.sample.ndim-1)))
            std[np.nonzero(std == 0.0)] = 1.0  # nan is because of dividing by zero
        self.mean = mean
        self.std = std

        print("found normalizing values")
        # self.features = (features - self.mean) / (2 * self.std) # Normalize instead using torchvision transforms

        self.transforms = torchvision.transforms.Compose([
            # torchvision.transforms.ToPILImage(),
            # torchvision.transforms.Resize((128, 128), Image.LINEAR),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std)
        ])

    def __getitem__(self, indices):
        target_idx = indices[0]
        distractors_idxs = indices[1:]

        distractors = []
        for d_idx in distractors_idxs:
            if self.use_different_targets:
                distractors.append(torch.stack(
                        (
                        self.transforms(self.pixels[d_idx, 0, :, :, :]),
                        self.transforms(self.pixels[d_idx, 1, :, :, :])
                        ), dim=0)
                    )
            else:
                distractors.append(self.transforms(self.pixels[d_idx]))

        if self.use_different_targets:
            target = torch.stack((
                    self.transforms(self.pixels[target_idx, 0, :, :, :]),
                    self.transforms(self.pixels[target_idx, 1, :, :, :])
                ), dim=0)
        else:
            target = self.transforms(self.pixels[target_idx])

        return (target, distractors, indices)

    def __len__(self):
        return self.pixels.shape[0]


class ImagesSampler(Sampler):
    def __init__(self, data_source, k, shuffle):
        self.n = len(data_source)
        self.k = k
        self.shuffle = shuffle
        assert self.k < self.n

    def __iter__(self):
        indices = []

        if self.shuffle:
            targets = torch.randperm(self.n).tolist()
        else:
            targets = list(range(self.n))

        for t in targets:
            arr = np.zeros(self.k + 1, dtype=int) # distractors + target
            arr[0] = t
            distractors = random.sample(range(self.n), self.k)
            while t in distractors:
                distractors = random.sample(range(self.n), self.k)
            arr[1:] = np.array(distractors)

            indices.append(arr)

        return iter(indices)

    def __len__(self):
        return self.n


class ImageFeaturesDataset():
    def __init__(self, features, mean=None, std=None):
        if mean is None:
            mean = np.mean(features, axis=0)
            std = np.std(features, axis=0)
            std[np.nonzero(std == 0.0)] = 1.0  # nan is because of dividing by zero
        self.mean = mean
        self.std = std
        self.features = (features - self.mean) / (2 * self.std)

    def __getitem__(self, indices):
        target_idx = indices[0]
        distractors_idxs = indices[1:]

        distractors = []
        for d_idx in distractors_idxs:
            distractors.append(self.features[d_idx])

        return (self.features[target_idx], distractors, indices)

    def __len__(self):
        return self.features.shape[0]


class ImageFeaturesDatasetZeroShot():
    def __init__(self, target_features, distractors_features, mean=None, std=None):
        if mean is None:
            mean = (np.mean(target_features, axis=0) + np.mean(distractors_features, axis=0)) / 2
            std = (np.std(target_features, axis=0) + np.std(distractors_features, axis=0)) / 2
            std[np.nonzero(std == 0.0)] = 1.0  # nan is because of dividing by zero
        self.mean = mean
        self.std = std

        self.target_features = (target_features - self.mean) / (2 * self.std)
        self.distractors_features = (distractors_features - self.mean) / (2 * self.std)

    def __getitem__(self, indices):
        target_idx = indices[0]
        distractors_idxs = indices[1:]

        distractors = []
        for d_idx in distractors_idxs:
            distractors.append(self.distractors_features[d_idx])

        return (self.target_features[target_idx], distractors, indices)

    def __len__(self):
        return self.target_features.shape[0]


class ImagesSamplerZeroShot(Sampler):
    def __init__(self, data_source, k, shuffle):
        self.n = len(data_source)
        self.k = k
        self.shuffle = shuffle
        assert self.k < self.n

    def __iter__(self):
        indices = []

        if self.shuffle:
            targets = torch.randperm(self.n).tolist()
        else:
            targets = list(range(self.n))

        for t in targets:
            arr = np.zeros(self.k + 1, dtype=int) # distractors + target
            arr[0] = t
            distractors = random.sample(range(self.n), self.k)
            # while t in distractors:
            #     distractors = random.sample(range(self.n), self.k)
            arr[1:] = np.array(distractors)

            indices.append(arr)

        return iter(indices)

    def __len__(self):
        return self.n
