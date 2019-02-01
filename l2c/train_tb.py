import os
import time
import pickle
import argparse
import torch as T
import numpy as np
from modules import Game
from utils import get_logger
from utils import AverageMeter
from utils import EarlyStopping
from utils import get_lr_scheduler
from torch.utils.data import DataLoader
from data_preprocessing import ImageDataset, ImagesSampler
from torch.utils.data.sampler import BatchSampler


use_gpu = T.cuda.is_available()
K = 3

def validate(valid_data, model, epoch, args, logger):
    batch_time_meter = AverageMeter()
    hinge_loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()
    entropy_meter = AverageMeter()

    logger.info("start validating...")
    model.eval()
    start = time.time()
    if use_gpu:
        device = T.device("cuda:{}".format(args.gpu_id))
    with T.no_grad():
        for d in valid_data:
            image_f, distractors = d
            if use_gpu:
                image_f = image_f.to(device=device)
                distractors = [d.to(device=device) for d in distractors]
                
            hinge_loss, accuracy, entropy = model(image_f, distractors, args.margin)

            hinge_loss_meter.update(hinge_loss.item())
            accuracy_meter.update(accuracy.item())
            entropy_meter.update(entropy.item())
            batch_time_meter.update(time.time() - start)
            start = time.time()

    logger.info(f"epoch: {epoch} hinge_loss: {hinge_loss_meter.avg:.4f} accuracy: {accuracy_meter.avg:.4f} "
                f"entropy: {entropy_meter.avg:.4f} batch_time: {batch_time_meter.avg:.4f}")

    return accuracy_meter.avg


def train(train_data, model, optimizer, epoch, args, logger):
    batch_time_meter = AverageMeter()
    hinge_loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()
    entropy_meter = AverageMeter()

    model.train()
    start = time.time()
    
    if use_gpu:
        device = T.device("cuda:{}".format(args.gpu_id))
    for image_f, distractors in train_data:
        if use_gpu:
            image_f = image_f.to(device=device)
            distractors = [d.to(device=device) for d in distractors]

        hinge_loss, accuracy, entropy = model(image_f, distractors, args.margin)
        optimizer.zero_grad()
        hinge_loss.backward()
        optimizer.step()

        hinge_loss_meter.update(hinge_loss.item())
        accuracy_meter.update(accuracy.item())
        entropy_meter.update(entropy.item())
        batch_time_meter.update(time.time() - start)

        del hinge_loss
        del accuracy
        del entropy
        start = time.time()

    logger.info(f"epoch: {epoch} hinge_loss: {hinge_loss_meter.avg:.4f} "
                f"accuracy: {accuracy_meter.avg:.4f} entropy: {entropy_meter.avg:.4f} "
                f"batch_time: {batch_time_meter.avg:.4f}")

    print('Epoch {}, loss {}, accuracy {}'.format(epoch, hinge_loss_meter.avg, accuracy_meter.avg))

train.prev_accuracy = -float("inf")


def main(args):
    print(hash(str(args)))
    if not os.path.exists(args.logs_path):
        os.makedirs(args.logs_path)
    logger = get_logger(f"{args.logs_path}/l{hash(str(args))}.log")
    logger.info(f"args: {str(args)}")
    logger.info(f"hash is: {hash(str(args))}")
    args.model_dir = f"{args.model_dir}/m{hash(str(args))}"
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    logger.info(f"checkpoint's dir is: {args.model_dir}")
    seed = hash(str(args)) % 1000_000
    T.manual_seed(seed)
    if use_gpu:
        T.cuda.manual_seed(seed)
    logger.info(f"random seed is: {seed}")

    logger.info("loading data...")
    with open("data/mscoco/dict.pckl", "rb") as f:
        d = pickle.load(f)
        word_to_idx = d["word_to_idx"]
        idx_to_word = d["idx_to_word"]
        bound_idx = word_to_idx["<S>"]
    train_features = np.load('data/mscoco/train_features.npy')
    valid_features = np.load('data/mscoco/valid_features.npy')
    logger.info('loaded...')

    args.vocab_size = len(word_to_idx)
    args.image_dim = valid_features.shape[1]


    train_dataset = ImageDataset(train_features)
    valid_dataset = ImageDataset(valid_features, mean=train_dataset.mean, std=train_dataset.std)

    # train_data = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    # valid_data = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True)

    train_data = DataLoader(train_dataset, num_workers=8, pin_memory=True, 
        batch_sampler=BatchSampler(ImagesSampler(train_dataset, K, shuffle=True), batch_size=args.batch_size, drop_last=True))

    valid_data = DataLoader(valid_dataset, num_workers=8, pin_memory=True,
        batch_sampler=BatchSampler(ImagesSampler(valid_dataset, K, shuffle=False), batch_size=args.batch_size, drop_last=True))


    model = Game(vocab_size=args.vocab_size, image_dim=args.image_dim,
                 s_embd_dim=args.sender_embd_dim, s_hid_dim=args.sender_hid_dim,
                 r_embd_dim=args.receiver_embd_dim, r_hid_dim=args.receiver_hid_dim,
                 bound_idx=bound_idx, max_steps=args.max_sentence_len, tau=args.tau,
                 straight_through=args.straight_through)
    if use_gpu:
        model = model.cuda(args.gpu_id)

    optimizer = T.optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = get_lr_scheduler(logger, optimizer)
    es = EarlyStopping(mode="max", patience=30, threshold=0.005, threshold_mode="rel")

    validate(valid_data, model, 0, args, logger)
    for epoch in range(args.max_epoch):
        train(train_data, model, optimizer, epoch, args, logger)
        val_accuracy = validate(valid_data, model, epoch, args, logger)
        if val_accuracy > train.prev_accuracy:
            logger.info("saving model...")
            T.save({"epoch": epoch, "state_dict": model.state_dict()}, f"{args.model_dir}/{epoch}.mdl")
            train.prev_accuracy = val_accuracy
        logger.info(90 * '=')
        lr_scheduler.step(val_accuracy)
        es.step(val_accuracy)
        if es.is_converged:
            logger.info("done")
            break



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-sentence-len", default=13, type=int)  # n + 1 (with one special token (<S>) at the beginning)
    parser.add_argument("--sender-embd-dim", default=256, type=int)
    parser.add_argument("--sender-hid-dim", default=512, type=int)
    parser.add_argument("--receiver-embd-dim", default=256, type=int)
    parser.add_argument("--receiver-hid-dim", default=512, type=int)
    parser.add_argument("--tau", default=1.2, type=float)
    parser.add_argument("--straight-through", default=True, type=lambda string: True if string == "True" else False)
    parser.add_argument("--margin", default=1.0, type=float)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--max-epoch", default=1000, type=int)
    parser.add_argument("--gpu-id", default=1, type=int)
    parser.add_argument("--model-dir", default="data/models/tb", type=str)
    parser.add_argument("--logs-path", required=False, default="data/logs/tb", type=str)
    main(parser.parse_args())
