from argparse import ArgumentParser

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import TrainDataset
from models import GMF
from models import MF
from models import MLP
from models import NCF
from neural_collaborative_filtering.Dataset import Dataset
from neural_collaborative_filtering.evaluate import evaluate_model


def train(model, train_dataloader, test_ratings, test_negatives, optimizer, n_epoch):
    for epoch in range(n_epoch):
        train_epoch(train_dataloader, model, optimizer)
        evaluate(model, test_ratings, test_negatives, epoch, K=10)


def train_epoch(train_dataloader, model, optimizer):
    for batch in train_dataloader:
        loss = model.compute_loss(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def evaluate(model, test_ratings, test_negatives, epoch, K=10):
    hits, ndcgs = evaluate_model(model, test_ratings, test_negatives, K=K, num_thread=1)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Epoch %4d:\t HR=%.4f, NDCG=%.4f\t' % (epoch + 1, hr, ndcg))


def get_model(args, n_users, n_items, device):
    if args.mf_n_components:
        model = MF(n_users, n_items, args.mf_n_components, device, args.loss_reduction)
    elif args.mlp_n_components and args.gmf_n_components:
        if args.mlp_filepath and args.gmf_filepath:
            model = NCF(n_users, n_items, args.mlp_n_components, args.gmf_n_components,
                        device, args.loss_reduction, args.mlp_filepath, args.gmf_filepath)
        else:
            model = NCF(n_users, n_items, args.mlp_n_components, args.gmf_n_components, device, args.loss_reduction)
    elif args.mlp_n_components:
        model = MLP(n_users, n_items, args.mlp_n_components, device, args.loss_reduction)
    else:
        model = GMF(n_users, n_items, args.gmf_n_components, device, args.loss_reduction)

    return model.to(device)


def get_optimizer(optimizer_name, parameters, lr, weight_decay):
    optimizer = optim.Adam if optimizer_name == 'Adam' else optim.SGD
    return optimizer(parameters, lr=lr, weight_decay=weight_decay)


def main(args):
    dataset = Dataset(args.dataset_path)

    train_positive_pairs = np.column_stack(dataset.trainMatrix.nonzero())
    train_dataset = TrainDataset(train_positive_pairs, dataset.num_items, n_negatives=args.n_negatives)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    test_ratings, test_negatives = (dataset.testRatings, dataset.testNegatives)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = get_model(args, dataset.num_users, dataset.num_items, device)
    optimizer = get_optimizer(args.optimizer, model.parameters(), args.lr, args.weight_decay)

    train(model, train_dataloader, test_ratings, test_negatives, optimizer, args.n_epoch)
    torch.save(model.state_dict(), args.checkpoint_name)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument(
        '--dataset_path',
        type=str,
        default='neural_collaborative_filtering/Data/ml-1m'
    )
    parser.add_argument(
        '--checkpoint_name',
        type=str,
        default='model.pth'
    )

    parser.add_argument(
        '--mf_n_components',
        type=int,
        default=None
    )
    parser.add_argument(
        '--mlp_n_components',
        type=int,
        default=None
    )
    parser.add_argument(
        '--gmf_n_components',
        type=int,
        default=None
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=4096
    )
    parser.add_argument(
        '--n_epoch',
        type=int,
        default=256
    )
    parser.add_argument(
        '--n_negatives',
        type=int,
        default=8
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.005
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        default='Adam',
        choices=('Adam', 'SGD')
    )
    parser.add_argument(
        '--loss_reduction',
        type=str,
        default='mean',
        choices=('mean', 'sum')
    )

    arguments = parser.parse_args()

    if not arguments.mf_n_components and not arguments.mlp_n_components and not arguments.gmf_n_components:
        parser.error('At least one of the following arguments must be present:'
                     ' mf_n_components, mlp_n_components, gmf_n_components.')

    main(arguments)
