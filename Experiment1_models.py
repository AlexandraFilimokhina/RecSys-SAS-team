from abc import ABC

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import TestDataset


class Recommender(nn.Module, ABC):
    def __init__(self, device, reduction):
        super().__init__()
        self.device = device
        self.reduction = reduction

    def compute_loss(self, batch):
        user_emb, item_emb, y_true = (item.to(self.device) for item in batch)
        y_pred = self(user_emb, item_emb)
        return F.binary_cross_entropy_with_logits(y_pred, y_true.float().view(y_pred.shape), reduction=self.reduction)

    @torch.no_grad()
    def predict(self, users_items, batch_size, verbose):
        del batch_size
        del verbose
        predictions = []
        test_dataloader = DataLoader(TestDataset(users_items), batch_size=4096, shuffle=False, pin_memory=True)
        for user_emb, item_emb in test_dataloader:
            user_emb, item_emb = user_emb.to(self.device), item_emb.to(self.device)
            prediction = self(user_emb, item_emb)
            predictions.extend(prediction.squeeze().cpu().tolist())
        return np.array(predictions)


class MF(Recommender):
    def __init__(self, n_users, n_items, n_components, device, reduction='sum'):
        super().__init__(device, reduction)

        self.user_emb = torch.nn.Embedding(n_users, n_components)
        self.item_emb = torch.nn.Embedding(n_items, n_components)

        self.user_emb.weight.data.normal_(0, 0.1)
        self.item_emb.weight.data.normal_(0, 0.1)

        self.user_bias = torch.nn.Embedding(n_users, 1)
        self.item_bias = torch.nn.Embedding(n_items, 1)

        self.user_bias.weight.data = torch.zeros(n_users, 1, dtype=torch.float)
        self.item_bias.weight.data = torch.zeros(n_items, 1, dtype=torch.float)

        self.mu = nn.Parameter(torch.zeros(1, dtype=torch.float))

    def forward(self, user_indices, item_indices):
        user_emb = self.user_emb(user_indices)
        item_emb = self.item_emb(item_indices)

        user_bias = self.user_bias(user_indices)
        item_bias = self.item_bias(item_indices)

        return torch.mul(user_emb, item_emb).sum(dim=1) + user_bias.view(-1) + item_bias.view(-1) + self.mu


class MLP(Recommender):
    def __init__(self, n_users, n_items, n_components, device, reduction='mean'):
        super().__init__(device, reduction)

        self.user_emb = nn.Embedding(num_embeddings=n_users, embedding_dim=n_components)
        self.item_emb = nn.Embedding(num_embeddings=n_items, embedding_dim=n_components)

        self.user_emb.weight.data.normal_(0, 0.1)
        self.item_emb.weight.data.normal_(0, 0.1)

        self.mlp = nn.Sequential(
            nn.Linear(2 * n_components, 2 * n_components),
            nn.ReLU(),
            nn.Linear(2 * n_components, n_components),
            nn.ReLU(),
            nn.Linear(n_components, n_components // 2),
            nn.ReLU()
        )

        self.clf = nn.Linear(n_components // 2, 1)

    def forward(self, user_emb, item_emb):
        return self.clf(self.mlp(torch.cat([self.user_emb(user_emb), self.item_emb(item_emb)], dim=-1)))

    def part_forward(self, user_emb, item_emb):
        return self.mlp(torch.cat([self.user_emb(user_emb), self.item_emb(item_emb)], dim=-1))


class GMF(Recommender):
    def __init__(self, n_users, n_items, n_components, device, reduction='mean'):
        super().__init__(device, reduction)

        self.user_emb = nn.Embedding(num_embeddings=n_users, embedding_dim=n_components)
        self.item_emb = nn.Embedding(num_embeddings=n_items, embedding_dim=n_components)

        self.user_emb.weight.data.normal_(0, 0.1)
        self.item_emb.weight.data.normal_(0, 0.1)

        self.clf = nn.Linear(n_components, 1)

    def forward(self, user_emb, item_emb):
        return self.clf(torch.mul(self.user_emb(user_emb), self.item_emb(item_emb)))

    def part_forward(self, user_emb, item_emb):
        return torch.mul(self.user_emb(user_emb), self.item_emb(item_emb))


class NCF(Recommender):
    def __init__(self, n_users, n_items, mlp_n_components, gmf_n_components, device, reduction='mean', mlp_filepath='', gmf_filepath=''):
        super().__init__(device, reduction)

        self.mlp = MLP(n_users, n_items, mlp_n_components, device)
        self.gmf = GMF(n_users, n_items, gmf_n_components, device)

        if mlp_filepath and gmf_filepath:
            self.mlp.load_state_dict(torch.load(mlp_filepath))
            self.gmf.load_state_dict(torch.load(gmf_filepath))

        self.clf = nn.Linear(gmf_n_components + mlp_n_components // 2, 1)

    def forward(self, user_emb, item_emb):
        return self.clf(torch.cat((self.mlp.part_forward(user_emb, item_emb), self.gmf.part_forward(user_emb, item_emb)), dim=-1))