import numpy as np
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, positive_pairs, n_items, n_negatives):
        self.positive_pairs = positive_pairs
        self.n_items = n_items
        self.n_negatives = n_negatives

        self.training_matrix = self._new_training_matrix()
        self.current = 0

    def __getitem__(self, idx):
        self.current += 1
        if self.current == len(self) + 1:
            self.current = 1
            self.training_matrix = self._new_training_matrix()
        return tuple(self.training_matrix[idx])

    def __len__(self):
        return len(self.training_matrix)

    def _new_training_matrix(self):
        training_matrix = np.empty([len(self.positive_pairs) * (1 + self.n_negatives), 3], dtype=np.int64)
        training_matrix[:len(self.positive_pairs), :2] = self.positive_pairs
        training_matrix[:len(self.positive_pairs), 2] = 1

        training_matrix[len(self.positive_pairs):, 0] = self.positive_pairs[:, 0].repeat(self.n_negatives)
        training_matrix[len(self.positive_pairs):, 1] = np.random.randint(0, self.n_items, len(self.positive_pairs) * self.n_negatives)
        training_matrix[len(self.positive_pairs):, 2] = 0

        return training_matrix


class TestDataset(Dataset):
    def __init__(self, pairs):
        self.users = pairs[0].astype(np.int64)
        self.items = pairs[1].astype(np.int64)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx]

    def __len__(self):
        return len(self.users)
