import torch as T
from torch import nn
from modules import Sender
from modules import Receiver

class Game(nn.Module):
    def __init__(self, vocab_size, image_dim, s_embd_dim, s_hid_dim, r_embd_dim, r_hid_dim, bound_idx, max_steps, tau, straight_through):
        super().__init__()
        self.sender = Sender(vocab_size, s_embd_dim, s_hid_dim, image_dim, bound_idx, max_steps, tau, straight_through)
        self.receiver = Receiver(vocab_size, r_embd_dim, r_hid_dim, image_dim)

    def forward(self, image_f, distractors, margin, greedy=False):
        device = image_f.device
        batch_size = image_f.shape[0]

        entropy, x = self.sender(image_f, greedy)

        predicted_f = self.receiver(x)

        image_f = image_f.view(batch_size, 1, -1)
        predicted_f = predicted_f.view(batch_size, -1, 1)#.to(device=device)

        target_score = T.bmm(image_f, predicted_f).squeeze()
        
        distractors_scores = []

        loss = 0
        for d in distractors:
            d = d.view(batch_size, 1, -1)#.to(device=device)
            d_score = T.bmm(d, predicted_f).squeeze()
            distractors_scores.append(d_score)
            zero_tensor = T.tensor(0.0, device=device)

            loss += T.max(zero_tensor, margin - target_score + d_score)

        all_scores = T.zeros((batch_size, 1 + len(distractors)), device=device)
        all_scores[:,0] = target_score

        for i, score in enumerate(distractors_scores):
            all_scores[:,i+1] = score

        all_scores = T.exp(all_scores)

        _, max_idx = T.max(all_scores, 1)

        accuracy = max_idx == 0
        accuracy = accuracy.to(dtype=T.float32)


        return T.mean(loss), T.mean(accuracy), T.mean(entropy)

        # scores = T.matmul(image_f, predicted_f.permute(1, 0))

        # pred_idxs = T.argmax(scores, dim=1, keepdim=True)

        # diag_idxs = T.tensor([[i] for i in range(image_f.shape[0])], device=device)
        

        # accuracy = (pred_idxs == diag_idxs).to(dtype=T.float32)

        # hinge_loss = T.max(margin - T.diagonal(scores)[:, None] + scores, other=T.tensor(0.0, device=device))        

        # hinge_loss.scatter_(-1, diag_idxs, 0.0)

        # sum_hinge_loss = T.sum(hinge_loss, dim=1)

        ####sum_hinge_loss = sum_hinge_loss * -log_prob

        # return T.mean(sum_hinge_loss), T.mean(accuracy), T.mean(entropy)
