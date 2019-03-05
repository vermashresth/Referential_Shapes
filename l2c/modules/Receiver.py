import torch as T
from torch import nn


class Receiver(nn.Module):
    def __init__(self, vocab_size, embd_dim, hid_dim, image_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embd_dim = embd_dim
        self.hid_dim = hid_dim
        self.image_dim = image_dim

        self.embd = nn.Parameter(T.empty((vocab_size, embd_dim), dtype=T.float32))
        self.lstm = nn.LSTM(embd_dim, hid_dim, num_layers=1)
        self.linear_im = nn.Linear(hid_dim, image_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.embd, 0.0, 0.1)

        nn.init.normal_(self.linear_im.weight, 0, 0.1)
        nn.init.constant_(self.linear_im.bias, 0)

        nn.init.xavier_uniform_(self.lstm.weight_ih_l0)
        nn.init.orthogonal_(self.lstm.weight_hh_l0)
        nn.init.constant_(self.lstm.bias_ih_l0, val=0)
        # cuDNN bias order: https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnRNNMode_t
        # add some positive bias for the forget gates [b_i, b_f, b_o, b_g] = [0, 1, 0, 0]
        nn.init.constant_(self.lstm.bias_hh_l0, val=0)
        nn.init.constant_(self.lstm.bias_hh_l0[self.hid_dim:2 * self.hid_dim], val=1)

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = T.zeros((batch_size, self.hid_dim), dtype=T.float32, device=x.device)
        c0 = T.zeros((batch_size, self.hid_dim), dtype=T.float32, device=x.device)
        e = T.matmul(x, self.embd) if x.dtype == T.float32 else self.embd[x]
        h, _ = self.lstm(e.permute(1, 0, 2), (h0[None, ...], c0[None, ...]))
        return self.linear_im(h[-1])
