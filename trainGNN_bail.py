#%%

import ipdb
import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

import warnings
warnings.filterwarnings('ignore')

from utils import *
from models import *
from torch_geometric.nn import GCNConv, SAGEConv, GINConv
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from torch_geometric.utils import dropout_adj, convert
from aif360.sklearn.metrics import consistency_score as cs
from aif360.sklearn.metrics import generalized_entropy_error as gee
import torch.nn as nn
from torch_sparse import SparseTensor
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from torch_geometric.utils import dropout_adj, convert
from aif360.sklearn.metrics import consistency_score as cs
from aif360.sklearn.metrics import generalized_entropy_error as gee
import torch.nn as nn
from torch_sparse import SparseTensor
from torch.nn import  Module
from typing import Callable
import torch
from torch import Tensor
from torch.nn import Dropout, ModuleList, BatchNorm1d, Module
from torch_geometric.nn import Linear

class MLP(Module):
    """
    A multi-layer perceptron (MLP) model.
    This implementation handles 0-layer configurations as well.
    """

    def __init__(self, *,
                 output_dim: int,
                 hidden_dim: int = 16,
                 num_layers: int = 2,
                 dropout: float = 0.0,
                 activation_fn: Callable[[Tensor], Tensor] = torch.relu_,
                 batch_norm: bool = False,
                 plain_last: bool = True,
                 ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout_fn = Dropout(dropout, inplace=True)
        self.activation_fn = activation_fn
        self.plain_last = plain_last

        dimensions = [hidden_dim] * (num_layers - 1) + [output_dim] * (num_layers > 0)
        self.layers: list[Linear] = ModuleList([Linear(-1, dim) for dim in dimensions])

        num_bns = batch_norm * (num_layers - int(plain_last))
        self.bns: list[BatchNorm1d] = []
        if batch_norm:
            self.bns = ModuleList([BatchNorm1d(hidden_dim) for _ in range(num_bns)])

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < self.num_layers - self.plain_last:
                x = self.bns[i](x) if self.bns else x
                x = self.dropout_fn(x)
                x = self.activation_fn(x)
        return x

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

        for bn in self.bns:
            bn.reset_parameters()

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout: float = 0.00, encoder_layer: int = 2, num_class: int = 1, num_layers: int = 2,
                 activation_fn: Callable[[Tensor], Tensor] = torch.relu_, normalize: bool = True, batch_norm: bool = False):
        super(MLPClassifier, self).__init__()
        self.dropout_fn = nn.Dropout(p=dropout, inplace=True)
        self.activation_fn = activation_fn
        self.normalize = normalize
        self.bn = nn.BatchNorm1d(hidden_dim) if batch_norm else None
        self.encoder_mlp = MLP(hidden_dim=input_dim,
                               output_dim=hidden_dim,
                               num_layers=encoder_layer,
                               activation_fn=activation_fn,
                               dropout=dropout,
                               batch_norm=batch_norm,
                               plain_last=True
                               )
        self.classifier_mlp = MLP(output_dim=output_dim,
                                  hidden_dim=hidden_dim,
                                  num_layers=1,
                                  dropout=dropout,
                                  activation_fn=activation_fn,
                                  batch_norm=batch_norm,
                                  plain_last=True,
                                  )

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder_mlp(x)

        x = self.bn(x) if self.bn is not None else x
        x = self.dropout_fn(x)
        x = self.activation_fn(x)
        if self.normalize:
            x = F.normalize(x, p=2, dim=-1)
        x = self.classifier_mlp(x)
        return x
    def pre_emb(self, x: Tensor) -> Tensor:
        x = self.encoder_mlp(x)
        x = self.bn(x) if self.bn is not None else x
        x = self.dropout_fn(x)
        x = self.activation_fn(x)
        if self.normalize:
            x = F.normalize(x, p=2, dim=-1)
        return x



def del_sensitive(features,i):
    result = torch.cat((features[:, :i], features[:, i+1:]), dim=1)
    return result

"""


save data
"""
class Data:
    def __init__(self, edge_index, features, labels, idx_train, idx_val, idx_test, sens, sens_idx):
        self.edge_index = edge_index
        self.features = features
        self.labels = labels
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test
        self.sens = sens
        self.sens_idx = sens_idx



def run(args):
    """
    Load data
    """
    # Load bail dataset
    if args.dataset == 'bail':
        sens_attr = "WHITE"  # column number after feature process is 0
        sens_idx = 0
        predict_attr = "RECID"
        label_number = 100
        path_bail = "./datasets/bail"
        adj, features, labels, idx_train, idx_val, idx_test, sens = load_bail(args.dataset, sens_attr,
                                                                              predict_attr, path=path_bail,
                                                                              label_number=label_number,
                                                                              )
        norm_features = feature_norm(features)
        norm_features[:, sens_idx] = features[:, sens_idx]
        features = norm_features
        features = del_sensitive(features, sens_idx)
        model_mlp = MLPClassifier(input_dim=features.shape[0], hidden_dim=args.unit, output_dim=1

                                  ).cuda()
        optimizer_mlp = optim.Adam(model_mlp.parameters(), lr=0.001, weight_decay=1e-3)

        # idx_train_mlp = label_idx[:min(int(0.55 * len(label_idx)), label_number)]
        #
        features=features.cuda()
        labels=labels.cuda()
        criterion_bce = torch.nn.BCEWithLogitsLoss()
        for epoch in range(500):
            t = time.time()
            model_mlp.train()
            optimizer_mlp.zero_grad()
            output = model_mlp(features)
            loss = criterion_bce(output[idx_train], labels[idx_train].unsqueeze(1).float())
            # Binary Cross-Entropy
            output_preds = (output.squeeze() > 0).type_as(labels)

            acc_test = accuracy_score(labels.cpu().numpy(), output_preds.cpu().numpy())

            loss.backward()
            optimizer_mlp.step()

        with torch.no_grad():
            output = model_mlp.pre_emb(features)
            features = output.detach().clone()
        # trained_gnn = GCN(nfeat=features.shape[1],
        #                   nhid=args.hidden,
        #                   nclass=1,
        #                   dropout=args.dropout)
        # trained_gnn = trained_gnn.to(args.device)
        # criterion_bce = torch.nn.BCEWithLogitsLoss()
        # edge_index = convert.from_scipy_sparse_matrix(adj)[0]
        # features = features.to(args.device)
        # num_nodes = features.shape[0]
        # edge_index = SparseTensor.from_edge_index(edge_index, sparse_sizes=(num_nodes, num_nodes), ).to(args.device)
        # label_idx = np.where(labels.cpu() >= 0)[0]
        # random.shuffle(label_idx)
        # labels = labels.to(args.device)
        # data = Data(edge_index, features, labels, idx_train, idx_val, idx_test, sens, sens_idx)
        # optimizer_van = optim.Adam(trained_gnn.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # train_vanilla(trained_gnn, optimizer_van, criterion_bce, args.epochs, data,
        #               save_name=f'{args.model}_llvanilla.pt')
        # trained_gnn.load_state_dict(torch.load(f'{args.model}_llvanilla.pt'))
        # # idx_train_mlp = label_idx[:min(int(0.55 * len(label_idx)), label_number)]
        # #
        # # features = features.cuda()
        # # labels = labels.cuda()
        # # criterion_bce = torch.nn.BCEWithLogitsLoss()
        # # for epoch in range(500):
        # #     t = time.time()
        # #     model_mlp.train()
        # #     optimizer_mlp.zero_grad()
        # #     output = model_mlp(features)
        # #     loss = criterion_bce(output[idx_train], labels[idx_train].unsqueeze(1).float())
        # #     # Binary Cross-Entropy
        # #     output_preds = (output.squeeze() > 0).type_as(labels)
        # #
        # #     acc_test = accuracy_score(labels.cpu().numpy(), output_preds.cpu().numpy())
        # #
        # #     loss.backward()
        # #     optimizer_mlp.step()
        # #
        # with torch.no_grad():
        #     output, _ = trained_gnn(features, edge_index)
        #     features = output.detach().clone()

        # load pokec dataset
    elif args.dataset == 'credit':
        sens_attr = "Age"  # column number after feature process is 1
        sens_idx = 1
        predict_attr = 'NoDefaultNextMonth'
        label_number = 6000
        path_credit = "./dataset/credit"
        adj, features, labels, idx_train, idx_val, idx_test, sens = load_credit(args.dataset, sens_attr,
                                                                                predict_attr, path=path_credit,
                                                                                label_number=label_number
                                                                                )
        norm_features = feature_norm(features)
        norm_features[:, sens_idx] = features[:, sens_idx]
        features = norm_features
        features = del_sensitive(features, sens_idx)
        model_mlp = MLPClassifier(input_dim=features.shape[0], hidden_dim=args.unit, output_dim=1

                                  ).cuda()
        optimizer_mlp = optim.Adam(model_mlp.parameters(), lr=0.001, weight_decay=1e-3)

        # idx_train_mlp = label_idx[:min(int(0.55 * len(label_idx)), label_number)]
        #
        features = features.cuda()
        labels = labels.cuda()
        criterion_bce = torch.nn.BCEWithLogitsLoss()
        for epoch in range(500):
            t = time.time()
            model_mlp.train()
            optimizer_mlp.zero_grad()
            output = model_mlp(features)
            loss = criterion_bce(output[idx_train], labels[idx_train].unsqueeze(1).float())
            # Binary Cross-Entropy
            output_preds = (output.squeeze() > 0).type_as(labels)

            acc_test = accuracy_score(labels.cpu().numpy(), output_preds.cpu().numpy())

            loss.backward()
            optimizer_mlp.step()

        with torch.no_grad():
            output = model_mlp.pre_emb(features)
            features = output.detach().clone()

    elif args.dataset == 'pokec_z':
        dataset = 'region_job'
        sens_attr = "region"
        predict_attr = "I_am_working_in_field"
        label_number = 4000
        sens_number = 200
        sens_idx = 3
        seed = 20
        path = "./datasets/pokec/"
        test_idx = False
        adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train = load_pokec(dataset,
                                                                                               sens_attr,
                                                                                               predict_attr,
                                                                                               path=path,
                                                                                               label_number=label_number,
                                                                                               sens_number=sens_number,
                                                                                               seed=seed,
                                                                                               test_idx=test_idx)
        labels[labels > 1] = 1
        features = del_sensitive(features, sens_idx)
        trained_gnn = SAGE(nfeat=features.shape[1],
                          nhid=args.unit,
                          nclass=1,
                          dropout=args.dropout)
        trained_gnn = trained_gnn.to(args.device)
        criterion_bce = torch.nn.BCEWithLogitsLoss()
        edge_index = convert.from_scipy_sparse_matrix(adj)[0]
        features = features.to(args.device)
        num_nodes = features.shape[0]
        edge_index = SparseTensor.from_edge_index(edge_index, sparse_sizes=(num_nodes, num_nodes), ).to(args.device)
        label_idx = np.where(labels.cpu() >= 0)[0]
        random.shuffle(label_idx)
        labels = labels.to(args.device)
        data = Data(edge_index, features, labels, idx_train, idx_val, idx_test, sens, sens_idx)
        optimizer_van = optim.Adam(trained_gnn.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        train_vanilla(trained_gnn, optimizer_van, criterion_bce, args.epochs, data,
                      save_name=f'{args.model}_llvanilla.pt')
        trained_gnn.load_state_dict(torch.load(f'{args.model}_llvanilla.pt'))
        # idx_train_mlp = label_idx[:min(int(0.55 * len(label_idx)), label_number)]
        #
        # features = features.cuda()
        # labels = labels.cuda()
        # criterion_bce = torch.nn.BCEWithLogitsLoss()
        # for epoch in range(500):
        #     t = time.time()
        #     model_mlp.train()
        #     optimizer_mlp.zero_grad()
        #     output = model_mlp(features)
        #     loss = criterion_bce(output[idx_train], labels[idx_train].unsqueeze(1).float())
        #     # Binary Cross-Entropy
        #     output_preds = (output.squeeze() > 0).type_as(labels)
        #
        #     acc_test = accuracy_score(labels.cpu().numpy(), output_preds.cpu().numpy())
        #
        #     loss.backward()
        #     optimizer_mlp.step()
        #
        with torch.no_grad():
            output, _ = trained_gnn(features, edge_index)
            features = output.detach().clone()

    elif args.dataset == 'pokec_n':
        dataset = 'region_job_2'
        sens_attr = "region"
        predict_attr = "I_am_working_in_field"
        label_number = 3500
        sens_number = 200
        sens_idx = 3
        seed = 20
        path = "./datasets/pokec/"
        test_idx = False
        adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train = load_pokec(dataset,
                                                                                               sens_attr,
                                                                                               predict_attr,
                                                                                               path=path,
                                                                                               label_number=label_number,
                                                                                               sens_number=sens_number,
                                                                                               seed=seed,
                                                                                               test_idx=test_idx)

        labels[labels > 1] = 1
        features=del_sensitive(features,sens_idx)
        trained_gnn = SAGE(nfeat=features.shape[1],
                          nhid=args.unit,
                          nclass=1,
                          dropout=args.dropout)
        trained_gnn = trained_gnn.to(args.device)
        criterion_bce = torch.nn.BCEWithLogitsLoss()
        edge_index = convert.from_scipy_sparse_matrix(adj)[0]
        features = features.to(args.device)
        num_nodes = features.shape[0]
        edge_index = SparseTensor.from_edge_index(edge_index, sparse_sizes=(num_nodes, num_nodes), ).to(args.device)
        label_idx = np.where(labels.cpu() >= 0)[0]
        random.shuffle(label_idx)
        labels = labels.to(args.device)
        data = Data(edge_index, features, labels, idx_train, idx_val, idx_test, sens, sens_idx)
        optimizer_van = optim.Adam(trained_gnn.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        train_vanilla(trained_gnn, optimizer_van, criterion_bce, args.epochs, data,
                      save_name=f'{args.model}_lvanilla.pt')
        trained_gnn.load_state_dict(torch.load(f'{args.model}_lvanilla.pt'))

        with torch.no_grad():
            output, _ = trained_gnn(features, edge_index)
            features = output.detach().clone()

    elif args.dataset == 'sport':
        dataset = 'sport'
        sens_idx = 0
        predict_attr = "sport"
        label_number = 3508
        sens_attr = 'race'
        sens_number = 3508
        seed = 20
        path = "./dataset/sport"
        test_idx = False
        sens_idx = 0
        adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train = load_sport(dataset,
                                                                                               sens_attr,
                                                                                               predict_attr,
                                                                                               path=path,
                                                                                               label_number=label_number,
                                                                                               sens_number=sens_number,
                                                                                               seed=seed,
                                                                                               test_idx=test_idx)
        # features = feature_norm(features)
        features = del_sensitive(features, sens_idx)
        # trained_gnn = GCN(nfeat=features.shape[1],
        #                   nhid=args.hidden,
        #                   nclass=1,
        #                   dropout=args.dropout)
        # trained_gnn = trained_gnn.to(args.device)
        # criterion_bce = torch.nn.BCEWithLogitsLoss()
        # edge_index = convert.from_scipy_sparse_matrix(adj)[0]
        # features = features.to(args.device)
        # num_nodes = features.shape[0]
        # edge_index = SparseTensor.from_edge_index(edge_index, sparse_sizes=(num_nodes, num_nodes), ).to(args.device)
        # label_idx = np.where(labels.cpu() >= 0)[0]
        # random.shuffle(label_idx)
        # labels = labels.to(args.device)
        # data = Data(edge_index, features, labels, idx_train, idx_val, idx_test, sens, sens_idx)
        # optimizer_van = optim.Adam(trained_gnn.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # train_vanilla(trained_gnn, optimizer_van, criterion_bce, args.epochs, data,
        #               save_name=f'{args.model}_llvanilla.pt')
        # trained_gnn.load_state_dict(torch.load(f'{args.model}_llvanilla.pt'))
        # # idx_train_mlp = label_idx[:min(int(0.55 * len(label_idx)), label_number)]
        # #
        # # features = features.cuda()
        # # labels = labels.cuda()
        # # criterion_bce = torch.nn.BCEWithLogitsLoss()
        # # for epoch in range(500):
        # #     t = time.time()
        # #     model_mlp.train()
        # #     optimizer_mlp.zero_grad()
        # #     output = model_mlp(features)
        # #     loss = criterion_bce(output[idx_train], labels[idx_train].unsqueeze(1).float())
        # #     # Binary Cross-Entropy
        # #     output_preds = (output.squeeze() > 0).type_as(labels)
        # #
        # #     acc_test = accuracy_score(labels.cpu().numpy(), output_preds.cpu().numpy())
        # #
        # #     loss.backward()
        # #     optimizer_mlp.step()
        # #
        # with torch.no_grad():
        #     output, _ = trained_gnn(features, edge_index)
        #     features = output.detach().clone()

    elif args.dataset == 'occupation':
        dataset = 'nba'
        sens_attr = "gender"
        predict_attr = "area"
        label_number = 6951
        sens_number = 6951
        seed = 20
        path = "./dataset/occupation"
        test_idx = False
        sens_idx = 0
        adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train = load_Occupation(dataset,
                                                                                                    sens_attr,
                                                                                                    predict_attr,
                                                                                                    path=path,
                                                                                                    label_number=label_number,
                                                                                                    sens_number=sens_number,
                                                                                                    seed=seed,
                                                                                                    test_idx=test_idx)
        features = feature_norm(features)
        features = del_sensitive(features, sens_idx)
        model_mlp = MLPClassifier(input_dim=features.shape[0], hidden_dim=args.unit, output_dim=1

                                  ).cuda()
        optimizer_mlp = optim.Adam(model_mlp.parameters(), lr=0.001, weight_decay=1e-3)
        trained_gnn = GCN(nfeat=features.shape[1],
                          nhid=args.unit,
                          nclass=1,
                          dropout=args.dropout)
        trained_gnn = trained_gnn.to(args.device)
        criterion_bce = torch.nn.BCEWithLogitsLoss()
        edge_index = convert.from_scipy_sparse_matrix(adj)[0]
        features = features.to(args.device)
        num_nodes = features.shape[0]
        edge_index = SparseTensor.from_edge_index(edge_index, sparse_sizes=(num_nodes, num_nodes), ).to(args.device)
        label_idx = np.where(labels.cpu() >= 0)[0]
        random.shuffle(label_idx)
        labels = labels.to(args.device)
        data = Data(edge_index, features, labels, idx_train, idx_val, idx_test, sens, sens_idx)
        optimizer_van = optim.Adam(trained_gnn.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        train_vanilla(trained_gnn, optimizer_van, criterion_bce, args.epochs, data,
                      save_name=f'{args.model}_llvanilla.pt')
        trained_gnn.load_state_dict(torch.load(f'{args.model}_llvanilla.pt'))
        # # idx_train_mlp = label_idx[:min(int(0.55 * len(label_idx)), label_number)]
        # #
        # # features = features.cuda()
        # # labels = labels.cuda()
        # # criterion_bce = torch.nn.BCEWithLogitsLoss()
        # # for epoch in range(500):
        # #     t = time.time()
        # #     model_mlp.train()
        # #     optimizer_mlp.zero_grad()
        # #     output = model_mlp(features)
        # #     loss = criterion_bce(output[idx_train], labels[idx_train].unsqueeze(1).float())
        # #     # Binary Cross-Entropy
        # #     output_preds = (output.squeeze() > 0).type_as(labels)
        # #
        # #     acc_test = accuracy_score(labels.cpu().numpy(), output_preds.cpu().numpy())
        # #
        # #     loss.backward()
        # #     optimizer_mlp.step()
        # #
        with torch.no_grad():
            output,_ = trained_gnn(features,edge_index)
            features = output.detach().clone()

    elif args.dataset =='nba':
        dataset = 'nba'
        sens_attr = "country"
        predict_attr = "SALARY"
        label_number = 100
        sens_number = 50
        seed = 20
        path = "./dataset/NBA"
        test_idx = True
        sens_idx= 1
        adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train = load_NBA(dataset,
                                                                                               sens_attr,
                                                                                               predict_attr,
                                                                                               path=path,
                                                                                               label_number=label_number,
                                                                                               sens_number=sens_number,
                                                                                               seed=seed,
                                                                                               test_idx=test_idx)
        features = feature_norm(features)
        features = del_sensitive(features, sens_idx)
        trained_gnn = GCN(nfeat=features.shape[1],
                          nhid=args.unit,
                          nclass=1,
                          dropout=args.dropout)
        trained_gnn = trained_gnn.to(args.device)
        criterion_bce = torch.nn.BCEWithLogitsLoss()
        edge_index = convert.from_scipy_sparse_matrix(adj)[0]
        features = features.to(args.device)
        num_nodes = features.shape[0]
        edge_index = SparseTensor.from_edge_index(edge_index, sparse_sizes=(num_nodes, num_nodes), ).to(args.device)
        label_idx = np.where(labels.cpu() >= 0)[0]
        random.shuffle(label_idx)
        labels = labels.to(args.device)
        # data = Data(edge_index, features, labels, idx_train, idx_val, idx_test, sens, sens_idx)
        # optimizer_van = optim.Adam(trained_gnn.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # train_vanilla(trained_gnn, optimizer_van, criterion_bce, args.epochs, data,
        #               save_name=f'{args.model}_llvanilla.pt')
        # trained_gnn.load_state_dict(torch.load(f'{args.model}_llvanilla.pt'))
        # # # idx_train_mlp = label_idx[:min(int(0.55 * len(label_idx)), label_number)]
        # # #
        # # # features = features.cuda()
        # # # labels = labels.cuda()
        # # # criterion_bce = torch.nn.BCEWithLogitsLoss()
        # # # for epoch in range(500):
        # # #     t = time.time()
        # # #     model_mlp.train()
        # # #     optimizer_mlp.zero_grad()
        # # #     output = model_mlp(features)
        # # #     loss = criterion_bce(output[idx_train], labels[idx_train].unsqueeze(1).float())
        # # #     # Binary Cross-Entropy
        # # #     output_preds = (output.squeeze() > 0).type_as(labels)
        # # #
        # # #     acc_test = accuracy_score(labels.cpu().numpy(), output_preds.cpu().numpy())
        # # #
        # # #     loss.backward()
        # # #     optimizer_mlp.step()
        # # #
        with torch.no_grad():
            output, _ = trained_gnn(features, edge_index)
            features = output.detach().clone()

    else:
        print('Invalid dataset name!!')
        exit(0)

    edge_index = convert.from_scipy_sparse_matrix(adj)[0]
    features = features.to(args.device)
    num_nodes = features.shape[0]
    edge_index = SparseTensor.from_edge_index(edge_index, sparse_sizes=(num_nodes, num_nodes), ).to(args.device)
    label_idx = np.where(labels.cpu() >= 0)[0]
    random.shuffle(label_idx)
    labels = labels.to(args.device)
    data = Data(edge_index, features, labels, idx_train, idx_val, idx_test, sens, sens_idx)


    """
    Train model (Synthetic teacher model and Student model)
    """

    num_class = 1

    """
    Build model and optimizer
    """


    # Student model and optimizer
    if args.model == 'gcn':


        # trained GNN model f_{cg} (for synthetic teacher)
        trained_gnn = GCN(nfeat=features.shape[1],
                          nhid=args.hidden,
                          nclass=num_class,
                          dropout=args.dropout)
        optimizer_van = optim.Adam(trained_gnn.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # optimizer_fair = optim.Adam(list(trained_gnn.body.parameters())+ list(trained_gnn.fc.parameters()), lr=args.lr, weight_decay=args.weight_decay)


        trained_gnn = trained_gnn.to(args.device)

    elif args.model=='sage':
        trained_gnn = SAGE(nfeat=features.shape[1],
                          nhid=args.hidden,
                          nclass=num_class,
                          dropout=args.dropout)
        optimizer_van = optim.Adam(trained_gnn.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    elif args.model =='gat':
        heads = ([1] * 1) + [1]
        trained_gnn = GAT(1, features.shape[1], args.hidden, heads, args.dropout, args.attn_drop,
                         0.2, False)
        optimizer_van = optim.Adam(trained_gnn.parameters(), lr=args.lr, weight_decay=args.weight_decay)


    elif args.model == 'gin':

        # trained GNN model f_{cg} (for synthetic teacher)
        trained_gnn = GIN(nfeat=features.shape[1],
                          nhid=args.hidden,
                          nclass=num_class,
                          dropout=args.dropout)
        optimizer_van = optim.Adam(trained_gnn.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # optimizer_fair = optim.Adam(list(trained_gnn.conv1.parameters()) + list(trained_gnn.fc.parameters())+ list(trained_gnn.mlp1.parameters()),

    elif args.model == 'jk':

        trained_gnn = JK(nfeat=features.shape[1],
                         nhid=args.hidden,
                         nclass=num_class,
                         dropout=args.dropout)
        optimizer_van = optim.Adam(trained_gnn.parameters(), lr=args.lr, weight_decay=args.weight_decay)


    trained_gnn = trained_gnn.to(args.device)

    """
    Train model (Synthetic teacher model and Student model)
    """
    criterion_bce = torch.nn.BCEWithLogitsLoss()
    criterion_cont = ContLoss(tem=args.tem)
    file_path=f'{args.model}_vanilla.pt'
    if os.path.exists(file_path):
        # 如果文件存在，删除它
        os.remove(file_path)
    #     print(f"{file_path} 已被删除")
    # else:
    #     print(f"{file_path} 不存在")
    # train f_{cg}
    train_vanilla(trained_gnn, optimizer_van, criterion_bce, args.epochs, data, save_name=f'{args.model}_lvanilla.pt')
    # for epoch in range(args.epochs):
    #     trained_gnn.train()
    #     optimizer.zero_grad()
    #
    #     h, output = trained_gnn(data.features, data.edge_index)
    #     loss_train = criterion(output[data.idx_train], data.labels[data.idx_train].unsqueeze(1).float())
    #     loss_train.backward()
    #     optimizer.step()

    # obtain node representations from the trained GNN model
    # trained_gnn.load_state_dict(torch.load(f'{args.model}_lvanilla.pt'))



    train_fair(trained_gnn, optimizer_van, criterion_bce, args.epochs_fair, data, save_name=f'{args.model}_lvanilla.pt',
               indice=args.indice, weight=args.weight,num=args.unit,name=args.dataset)

    auc, f1, acc, dp, eo = evaluation(trained_gnn, f'{args.model}_lvanilla.pt', data)
    print(auc)
    print(f1)
    print(acc)
    print(dp)
    print(eo)

    return auc, f1, acc, dp, eo


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--seed_num', type=int, default=0, help='The number of random seed.')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
    parser.add_argument('--proj_hidden', type=int, default=16,
                        help='Number of hidden units in the projection layer of encoder.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--dataset', type=str, default='loan',
                        choices=['nba', 'bail', 'pokec_z', 'pokec_n', 'credit', 'german','occupation','sport','credit'])
    parser.add_argument("--num_heads", type=int, default=1, help="number of hidden attention heads")
    parser.add_argument("--num_out_heads", type=int, default=1, help="number of output attention heads")
    parser.add_argument("--num_layers", type=int, default=2, help="number of hidden layers")
    parser.add_argument('--model', type=str, default='gcn', choices=['gcn', 'gin','sage','jk','gat'])
    parser.add_argument('--encoder', type=str, default='gcn')
    parser.add_argument('--tem', type=float, default=0.5, help='the temperature of contrastive learning loss '
                                                               '(mutual information maximize)')
    parser.add_argument('--gamma', type=float, default=0.25, help='empower coefficient')
    parser.add_argument('--lr_w', type=float, default=1,
                        help='the learning rate of the adaptive weight coefficient')
    parser.add_argument('--indice', type=int, default=10,
                        help='the learning rate of the adaptive weight coefficient')
    parser.add_argument('--weight', type=float, default=1,
                        help='the learning rate of the adaptive weight coefficient')
    parser.add_argument('--unit', type=int, default=16,
                        help='the learning rate of the adaptive weight coefficient')
    parser.add_argument('--epochs_fair', type=int, default=30,
                        help='the learning rate of the adaptive weight coefficient')
    args = parser.parse_known_args()[0]
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # set device
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    auc, f1, acc, dp, eo = np.zeros(shape=(args.seed_num, 2)), np.zeros(shape=(args.seed_num, 2)), \
                                     np.zeros(shape=(args.seed_num, 2)), np.zeros(shape=(args.seed_num, 2)), \
                                     np.zeros(shape=(args.seed_num, 2))

    for seed in range(args.seed_num):
        # set seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        if args.cuda:
            torch.cuda.manual_seed(seed)

        # torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        auc[seed, :], f1[seed, :], acc[seed, :], dp[seed, :], eo[seed, :] = run(args)

        print(f"========seed {seed}========")

    # print report
    print("=================START=================")
    print(f"Parameter:τ={args.tem}, γ={args.gamma}, lr_w={args.lr_w}")
    print(f"AUCROC: {np.around(np.mean(auc[:, 0]) * 100, 2)} ± {np.around(np.std(auc[:, 0]) * 100, 2)}")
    print(f'F1-score: {np.around(np.mean(f1[:, 0]) * 100, 2)} ± {np.around(np.std(f1[:, 0]) * 100, 2)}')
    print(f'ACC: {np.around(np.mean(acc[:, 0]) * 100, 2)} ± {np.around(np.std(acc[:, 0]) * 100, 2)}')
    print(f'parity: {np.around(np.mean(dp[:, 0]) * 100, 2)} ± {np.around(np.std(dp[:, 0]) * 100, 2)}')
    print(f'Equality: {np.around(np.mean(eo[:, 0]) * 100, 2)} ± {np.around(np.std(eo[:, 0]) * 100, 2)}')
    print("=================END=================")