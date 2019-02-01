import pickle
import torch as T
import numpy as np
from modules import Game
from utils import AverageMeter
from torch.utils.data import DataLoader
from data_preprocessing import ImageDataset


with open("data/mscoco/dict.pckl", "rb") as f:
    d = pickle.load(f)
    word_to_idx = d["word_to_idx"]
    idx_to_word = d["idx_to_word"]
    bound_idx = word_to_idx["<S>"]
train_features = np.load('data/mscoco/train_features.npy')
test_features = np.load('data/mscoco/test_features.npy')

train_dataset = ImageDataset(train_features)
test_dataset = ImageDataset(test_features, mean=train_dataset.mean, std=train_dataset.std)


test_data = DataLoader(test_dataset, batch_size=128, num_workers=8, pin_memory=True)
print(len(test_dataset))

gpu_id = 0
device = T.device("cuda:{}".format(gpu_id))



model = Game(vocab_size=len(word_to_idx), image_dim=test_features.shape[1], s_embd_dim=256, s_hid_dim=512, r_embd_dim=256, r_hid_dim=512, bound_idx=bound_idx,
             max_steps=5, tau=1, straight_through=True).cuda(gpu_id)

checkpoint = T.load("data/models/tb/m5883228619855198343/119.mdl")
model.load_state_dict(checkpoint["state_dict"])
model.eval()

hinge_loss_meter = AverageMeter()
accuracy_meter = AverageMeter()
entropy_meter = AverageMeter()
for i, image_f in enumerate(test_data):
    image_f = image_f.to(device=device)

    model.eval()
    hinge_loss, accuracy, entropy = model(image_f, 1.0, greedy=True)
    hinge_loss_meter.update(hinge_loss.item())
    accuracy_meter.update(accuracy.item())
    entropy_meter.update(entropy.item())


print(accuracy_meter.avg)
print(hinge_loss_meter.avg)
print(entropy_meter.avg)
print(accuracy_meter.count)