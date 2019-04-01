import unittest
import torch

from model import Model


class TestModel(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.model = Model(n_image_features=4096, vocab_size=3,
            embedding_dim=256, hidden_size=512, batch_size=2, 
            bound_idx=2, max_sentence_length=5, use_gpu=False)

    def test_get_word_counts_train(self):
        self.model.train()

        m = torch.tensor([
                [
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                ],
                [
                    [0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                ]
            ])

        res = self.model._get_word_counts(m)

        self.assertTrue(torch.all(torch.eq(res, torch.tensor([3, 4, 5]))))

    def test_get_word_counts_eval(self):
        self.model.eval()

        m = torch.tensor([
                [2, 1, 0, 0, 2, 2],
                [2, 0, 1, 0, 1, 0]
            ])

        res = self.model._get_word_counts(m)

        self.assertTrue(torch.all(torch.eq(res, torch.tensor([5, 3, 4]))))

    def test_pad_train(self):
        self.model.train()

        m = torch.tensor([
                [
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                [
                    [0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                ]
            ])

        seq_lengths = torch.tensor([4, 6])

        res = self.model._pad(m, seq_lengths)

        self.assertTrue(torch.all(
            torch.eq(
                res,
                torch.tensor([
                    [
                        [0.0, 0.0, 1.0],
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                        [0.0, 0.0, 1.0],
                        [0.0, 0.0, 1.0],
                    ],
                    [
                        [0.0, 0.0, 1.0],
                        [0.0, 1.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 1.0, 0.0],
                    ]
                ])
            )))

    def test_pad_eval(self):
        self.model.eval()

        m = torch.tensor([
                [2, 1, 0, 0, 2, 0],
                [2, 0, 1, 0, 1, 0]
            ])

        seq_lengths = torch.tensor([5, 6])

        res = self.model._pad(m, seq_lengths)

        self.assertTrue(torch.all(
            torch.eq(
                res,
                    torch.tensor([
                    [2, 1, 0, 0, 2, 2],
                    [2, 0, 1, 0, 1, 0]
                ])
            )))



    