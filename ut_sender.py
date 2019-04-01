import unittest
import torch

from model import Sender


class TestSender(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.sender = Sender(n_image_features=4096, vocab_size=3, 
            embedding_dim=256, hidden_size=512, batch_size=2, 
            bound_idx=2, max_sentence_length=3,
            use_gpu=False, greedy=True)

    def test_calculate_seq_len_train(self):
        self.sender.train()

        initial_length = self.sender.max_sentence_length + 1
        seq_lengths = torch.ones([self.sender.batch_size], dtype=torch.int64) * initial_length

        timestep = 1
        token = torch.tensor([
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]
            ])

        self.sender._calculate_seq_len(seq_lengths, token, initial_length, timestep)
        self.assertTrue(torch.all(torch.eq(seq_lengths, torch.tensor([initial_length, 2]))))

        timestep = 2
        token = torch.tensor([
                [1.0, 0.0, 0.0],
                [0.1, 0.0, 0.0]
            ])

        self.sender._calculate_seq_len(seq_lengths, token, initial_length, timestep)
        self.assertTrue(torch.all(torch.eq(seq_lengths, torch.tensor([initial_length, 2]))))

        timestep = 3
        token = torch.tensor([
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0]
            ])

        self.sender._calculate_seq_len(seq_lengths, token, initial_length, timestep)
        self.assertTrue(torch.all(torch.eq(seq_lengths, torch.tensor([initial_length, 2]))))


    def test_calculate_seq_len_eval(self):
        self.sender.eval()

        initial_length = self.sender.max_sentence_length + 1
        seq_lengths = torch.ones([self.sender.batch_size], dtype=torch.int64) * initial_length

        timestep = 1
        token = torch.tensor([1, 2])

        self.sender._calculate_seq_len(seq_lengths, token, initial_length, timestep)
        self.assertTrue(torch.all(torch.eq(seq_lengths, torch.tensor([initial_length, 2]))))

        timestep = 2
        token = torch.tensor([2, 2])

        self.sender._calculate_seq_len(seq_lengths, token, initial_length, timestep)
        self.assertTrue(torch.all(torch.eq(seq_lengths, torch.tensor([3, 2]))))

        timestep = 3
        token = torch.tensor([2, 0])

        self.sender._calculate_seq_len(seq_lengths, token, initial_length, timestep)
        self.assertTrue(torch.all(torch.eq(seq_lengths, torch.tensor([3, 2]))))

    

    