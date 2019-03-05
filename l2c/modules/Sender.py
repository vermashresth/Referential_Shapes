import torch as T
from torch import nn
from torch.nn import functional as F
from torch.distributions.categorical import Categorical
# from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical


class Sender(nn.Module):
    def __init__(self, vocab_size, embd_dim, hid_dim, image_dim, bound_idx, max_steps, tau, straight_through):
        super().__init__()
        self.vocab_size = vocab_size
        self.embd_dim = embd_dim
        self.hid_dim = hid_dim
        self.image_dim = image_dim
        self.bound_idx = bound_idx
        self.max_steps = max_steps
        # self.tau = tau
        # self.straight_through = straight_through

        self.embd = nn.Embedding(vocab_size, embd_dim)#nn.Parameter(T.empty((vocab_size, embd_dim), dtype=T.float32))
        self.linear_im = nn.Linear(image_dim, hid_dim)
        self.linear_vocab = nn.Linear(hid_dim, vocab_size)
        self.lstm_cell = nn.LSTMCell(embd_dim, hid_dim)
        self.reset_parameters()

    def reset_parameters(self):
        # nn.init.normal_(self.embd, 0.0, 0.1)
        nn.init.normal_(self.embd.weight, 0.0, 0.1)

        nn.init.normal_(self.linear_im.weight, 0, 0.1)
        nn.init.constant_(self.linear_im.bias, 0)

        nn.init.constant_(self.linear_vocab.weight, 0)
        nn.init.constant_(self.linear_vocab.bias, 0)

        nn.init.xavier_uniform_(self.lstm_cell.weight_ih)
        nn.init.orthogonal_(self.lstm_cell.weight_hh)
        nn.init.constant_(self.lstm_cell.bias_ih, val=0)
        # cuDNN bias order: https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnRNNMode_t
        # add some positive bias for the forget gates [b_i, b_f, b_o, b_g] = [0, 1, 0, 0]
        nn.init.constant_(self.lstm_cell.bias_hh, val=0)
        nn.init.constant_(self.lstm_cell.bias_hh[self.hid_dim:2 * self.hid_dim], val=1)


    def forward(self, image_f, greedy=False):
        batch_size = image_f.shape[0]
        device = image_f.device

        entropy = 0.0
        #if self.training:
            # x = [T.zeros((image_f.shape[0], self.vocab_size), dtype=T.float32, device=image_f.device)]
            # x[0][:, self.bound_idx] = 1.0
            # x = [T.zeros((image_f.shape[0], ), dtype=T.float32, device=image_f.device)]
            # x[0] = x[0] + self.bound_idx
        # else:
        #     x = [T.full((image_f.shape[0], ), fill_value=self.bound_idx, dtype=T.int64, device=image_f.device)]


        h_t = self.linear_im(image_f)
        c_t = T.zeros_like(h_t)

        input = T.ones((batch_size), dtype=T.int64, device=device) * self.bound_idx
        e_t = self.embd(input)
        x = [input]

        for t in range(self.max_steps):
            # e_t = T.matmul(x[-1], self.embd) if x[-1].dtype == T.float32 else self.embd[x[-1]]
            h_t, c_t = self.lstm_cell(e_t, (h_t, c_t))
            scores = self.linear_vocab(h_t)
            probs_t = F.softmax(scores, dim=-1)
            cat = Categorical(probs_t)
            entropy += cat.entropy()
            if self.training:
                # cat_distr = RelaxedOneHotCategorical(self.tau, probs_t)
                x_t = cat.sample()
                # x_t = cat_distr.rsample()
                
                # if self.straight_through:
                #     x_t_hard = T.zeros_like(x_t)
                #     x_t_hard.scatter_(-1, T.argmax(x_t, dim=-1, keepdim=True), 1.0)
                #     x_t = (x_t_hard - x_t).detach() + x_t
            else:
                x_t = T.argmax(probs_t, dim=-1) if greedy else Categorical(probs_t).sample()
                
            x.append(x_t)
            e_t = self.embd(x_t)

        return T.mean(entropy) / self.max_steps, T.stack(x, dim=1), cat.log_prob(x_t)
