import os

import torch
import random
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import distance_matrix
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import torch.optim as optim


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def build_relationship(x, thresh=0.25):
    df_euclid = pd.DataFrame(1 / (1 + distance_matrix(x.T.T, x.T.T)), columns=x.T.columns, index=x.T.columns)
    df_euclid = df_euclid.to_numpy()
    idx_map = []
    for ind in range(df_euclid.shape[0]):
        max_sim = np.sort(df_euclid[ind, :])[-2]
        neig_id = np.where(df_euclid[ind, :] > thresh*max_sim)[0]
        import random
        random.seed(912)
        random.shuffle(neig_id)
        for neig in neig_id:
            if neig != ind:
                idx_map.append([ind, neig])
    # print('building edge relationship complete')
    idx_map =  np.array(idx_map)
    
    return idx_map

def load_Occupation(dataset, sens_attr, predict_attr, path="/root/autodl-tmp/project/FairGKD-master/datasets/NBA", label_number=1000, sens_number=500,
               seed=19, test_idx=False):
    idx_features_labels = pd.read_csv('/root/autodl-tmp/project/FairGKD-master/datasets/occupation/occupation.csv')
    header = list(idx_features_labels.columns)

    # header.remove(predict_attr)
    header.remove('user_id')
    header.remove('embeddings')
    # header.remove(sens_attr)
    header.remove(predict_attr)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values

    # build graph
    idx = np.array(idx_features_labels["user_id"], dtype=int)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(os.path.join('/root/autodl-tmp/project/FairGKD-master/datasets/occupation/occupation_edges.txt'), dtype=int)

    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # features = normalize(features)
    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    idx = np.array(idx_features_labels["user_id"], dtype=int)
    idx_map = {j: i for i, j in enumerate(idx)}
    import random
    random.seed(seed)
    label_idx = np.where(labels >= 0)[0]
    random.shuffle(label_idx)
    n = len(label_idx)
    train_ratio = 0.5
    idx_train = label_idx[:int(n * train_ratio)]
    idx_val = label_idx[int(n * train_ratio): int(n * (1 + train_ratio) / 2)]
    idx_test = label_idx[int(n * (1 + train_ratio) / 2):]
    sens = idx_features_labels[sens_attr].values  # print(sens, type(sens))
    sens_idx = set(np.where(sens >= 0)[0])
    # idx_test = np.asarray(list(sens_idx & set(idx_test)))

    sens = torch.FloatTensor(sens)
    idx_sens_train = list(sens_idx - set(idx_val) - set(idx_test))
    random.shuffle(idx_sens_train)
    idx_sens_train = torch.LongTensor(idx_sens_train[:sens_number])
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    return adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train


from torch_geometric.utils import dropout_adj, convert


def load_sport(dataset, sens_attr, predict_attr, path="/root/autodl-tmp/project/FairGKD-master/datasets/NBA", label_number=1000, sens_number=500,
               seed=19, test_idx=False):
    idx_features_labels = pd.read_csv('/root/autodl-tmp/project/FairGKD-master/datasets/sport/sport.csv')
    header = list(idx_features_labels.columns)

    # header.remove(predict_attr)
    header.remove('user_id')
    header.remove('embeddings')
    # header.remove(sens_attr)
    header.remove(predict_attr)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    features = torch.FloatTensor(np.array(features.todense()))
    labels = idx_features_labels[predict_attr].values

    # build graph
    idx = np.array(idx_features_labels["user_id"], dtype=int)
    idx_map = {j: i for i, j in enumerate(idx)}
    label_idx = np.where(labels >= 0)[0]
    random.shuffle(label_idx)
    n = len(label_idx)
    train_ratio = 0.6
    idx_train = label_idx[:int(n * train_ratio)]
    idx_val = label_idx[int(n * train_ratio): int(n * (1 + train_ratio) / 2)]
    idx_test = label_idx[int(n * (1 + train_ratio) / 2):]
    sens = idx_features_labels[sens_attr].values  # print(sens, type(sens))
    sens_idx = set(np.where(sens >= 0)[0])
    # idx_test = np.asarray(list(sens_idx & set(idx_test)))


    sens = torch.FloatTensor(sens)
    idx_sens_train = list(sens_idx - set(idx_val) - set(idx_test))
    random.shuffle(idx_sens_train)
    idx_sens_train = torch.LongTensor(idx_sens_train[:sens_number])
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    num_nodes=labels.shape[0]
    edges_unordered = np.genfromtxt(f'/root/autodl-tmp/project/FairGKD-master/datasets/sport/sport_edges.txt', dtype=int)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(num_nodes, num_nodes),
                        dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    labels = torch.LongTensor(labels)


    return adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train




def load_NBA(dataset, sens_attr, predict_attr, path="/root/autodl-tmp/project/FairGKD-master/datasets/NBA", label_number=1000, sens_number=500,
               seed=19, test_idx=False):
    idx_features_labels = pd.read_csv('/root/autodl-tmp/project/FairGKD-master/datasets/NBA/nba.csv')
    header = list(idx_features_labels.columns)
    header.remove("user_id")

    # header.remove(sens_attr)
    header.remove(predict_attr)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values

    # build graph
    idx = np.array(idx_features_labels["user_id"], dtype=int)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt('/root/autodl-tmp/project/FairGKD-master/datasets/NBA/nba_relationship.txt', dtype=int)

    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # features = normalize(features)
    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    import random
    random.seed(seed)
    label_idx = np.where(labels >= 0)[0]
    random.shuffle(label_idx)

    idx_train = label_idx[:min(int(0.5 * len(label_idx)), label_number)]
    idx_val = label_idx[int(0.5 * len(label_idx)):int(0.75 * len(label_idx))]
    if test_idx:
        idx_test = label_idx[label_number:]
        idx_val = idx_test
    else:
        idx_test = label_idx[int(0.75 * len(label_idx)):]

    sens = idx_features_labels[sens_attr].values

    sens_idx = set(np.where(sens >= 0)[0])
    sens = torch.FloatTensor(sens)
    idx_sens_train = list(sens_idx - set(idx_val) - set(idx_test))
    random.seed(seed)
    random.shuffle(idx_sens_train)
    idx_sens_train = torch.LongTensor(idx_sens_train[:sens_number])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train


def load_pokec(dataset, sens_attr, predict_attr, path="./datasets/pokec/", label_number=1000, sens_number=500,
               seed=19, test_idx=False):
    """Load data"""
    # print('Loading {} dataset from {}'.format(dataset, path))

    idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove("user_id")

    # header.remove(sens_attr)
    header.remove(predict_attr)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values

    # build graph
    idx = np.array(idx_features_labels["user_id"], dtype=int)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(os.path.join(path, "{}_relationship.txt".format(dataset)), dtype=int)

    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # features = normalize(features)
    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    import random
    random.seed(seed)
    label_idx = np.where(labels >= 0)[0]
    random.shuffle(label_idx)

    idx_train = label_idx[:min(int(0.5 * len(label_idx)), label_number)]
    idx_val = label_idx[int(0.5 * len(label_idx)):int(0.75 * len(label_idx))]
    if test_idx:
        idx_test = label_idx[label_number:]
        idx_val = idx_test
    else:
        idx_test = label_idx[int(0.75 * len(label_idx)):]

    sens = idx_features_labels[sens_attr].values

    sens_idx = set(np.where(sens >= 0)[0])
    idx_test = np.asarray(list(sens_idx & set(idx_test)))
    sens = torch.FloatTensor(sens)
    idx_sens_train = list(sens_idx - set(idx_val) - set(idx_test))
    random.seed(seed)
    random.shuffle(idx_sens_train)
    idx_sens_train = torch.LongTensor(idx_sens_train[:sens_number])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train


def load_bail(dataset, sens_attr="WHITE", predict_attr="RECID", path="./dataset/bail/", label_number=1000):
    # print('Loading {} dataset from {}'.format(dataset, path))
    idx_features_labels = pd.read_csv(os.path.join(path,"{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)
    # header.remove(sens_attr)
    
    # # Normalize School
    # idx_features_labels['SCHOOL'] = 2*(idx_features_labels['SCHOOL']-idx_features_labels['SCHOOL'].min()).div(idx_features_labels['SCHOOL'].max() - idx_features_labels['SCHOOL'].min()) - 1

    # # Normalize RULE
    # idx_features_labels['RULE'] = 2*(idx_features_labels['RULE']-idx_features_labels['RULE'].min()).div(idx_features_labels['RULE'].max() - idx_features_labels['RULE'].min()) - 1

    # # Normalize AGE
    # idx_features_labels['AGE'] = 2*(idx_features_labels['AGE']-idx_features_labels['AGE'].min()).div(idx_features_labels['AGE'].max() - idx_features_labels['AGE'].min()) - 1

    # # Normalize TSERVD
    # idx_features_labels['TSERVD'] = 2*(idx_features_labels['TSERVD']-idx_features_labels['TSERVD'].min()).div(idx_features_labels['TSERVD'].max() - idx_features_labels['TSERVD'].min()) - 1

    # # Normalize FOLLOW
    # idx_features_labels['FOLLOW'] = 2*(idx_features_labels['FOLLOW']-idx_features_labels['FOLLOW'].min()).div(idx_features_labels['FOLLOW'].max() - idx_features_labels['FOLLOW'].min()) - 1

    # # Normalize TIME
    # idx_features_labels['TIME'] = 2*(idx_features_labels['TIME']-idx_features_labels['TIME'].min()).div(idx_features_labels['TIME'].max() - idx_features_labels['TIME'].min()) - 1

    # build relationship
    if os.path.exists(f'{path}/{dataset}_edges.txt'):
        edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int')
    else:
        edges_unordered = build_relationship(idx_features_labels[header], thresh=0.6)
        np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values

    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # features = normalize(features)
    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    import random
    random.seed(20)
    label_idx_0 = np.where(labels==0)[0]
    label_idx_1 = np.where(labels==1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)
    idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number//2)], label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number//2)])
    idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))], label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])

    sens = idx_features_labels[sens_attr].values.astype(int)
    sens = torch.FloatTensor(sens)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    
    return adj, features, labels, idx_train, idx_val, idx_test, sens


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def feature_norm(features):
    min_values = features.min(axis=0)[0]
    max_values = features.max(axis=0)[0]
    return 2*(features - min_values).div(max_values-min_values) - 1

def find_indices(distance, select_adj, num_indices):
    distance = distance.clone()
    distance[select_adj] = float('inf')
    _, indices = torch.topk(distance, num_indices, largest=False)
    return indices

# def find_counterfactuals( embed, preds, sens, indices_num):
#     embed = embed.detach()
#     pseudo_label = preds.reshape(-1, 1).bool()
#     distance = torch.cdist(embed, embed, p=2)
#     distance = (distance - distance.min()) / (distance.max() - distance.min())
#     label_pair = pseudo_label.eq(pseudo_label.t())
#     _, indices = torch.topk(distance, 1000,  largest=False)
#     bool_matrix = torch.zeros_like(distance, dtype=torch.bool)
#     bool_matrix[torch.arange(indices.size(0)).unsqueeze(1), indices] = True
#     select_adj2 = ~label_pair&~bool_matrix
#     distance = torch.abs(sens.unsqueeze(0) - sens.unsqueeze(1))
#     distance = (distance - distance.min()) / (distance.max() - distance.min())
#     distance=distance*-1
#     indices2 = find_indices(distance.cpu(), select_adj2, indices_num)
#     return indices2


def find_counterfactuals(embed, preds, sens, indices_num):
    embed = embed.detach()

    # Compute label_pair and sens_pair
    pseudo_label = preds.reshape(-1, 1).bool()
    # sens_torch = sens.reshape(-1, 1).bool()
    label_pair = pseudo_label.eq(pseudo_label.t())
    # sens_pair = sens_torch.eq(sens_torch.t())
    distance = torch.abs(sens.unsqueeze(0) - sens.unsqueeze(1))
    distance = (distance - distance.min()) / (distance.max() - distance.min())
    _, indices = torch.topk(distance, int(sens.shape[0]/10), largest=True)
    bool_matrix = torch.zeros_like(distance, dtype=torch.bool)
    bool_matrix[torch.arange(indices.size(0)).unsqueeze(1), indices] = True
    # Compute select_adj and select_adj2
    select_adj = ~label_pair & ~bool_matrix


    # Compute the normalized distance
    distance = torch.cdist(embed, embed, p=2)
    distance = (distance - distance.min()) / (distance.max() - distance.min())

    # Find indices

    indices2 = find_indices(distance.cpu(), select_adj, indices_num)

    return indices2

def train_fair(model, optimizer, criterion, epochs, data, save_name,indice,weight,num,name):
    features=data.features
    edge_index=data.edge_index
    labels=data.labels
    idx_train=data.idx_train
    idx_complete_train=data.idx_train
    idx_complete_val=data.idx_val

    indices_num=indice
    idx_val=data.idx_val
    idx_test=data.idx_test
    # idx_complete_train = torch.tensor(
    #     list(set(torch.arange(0, features.shape[0]).tolist()) - set(idx_test.tolist()) - set(idx_val.tolist())))
    # idx_complete_val = torch.tensor(
    #     list(set(torch.arange(0, features.shape[0]).tolist()) - set(idx_test.tolist()) - set(idx_train.tolist())))
    weightSum =weight
    sens=data.sens

    all_indices = list(range(num))
    related_weights = []
    related_attrs = all_indices

    for zz in range(num):
        related_weights.append(1)

    best_loss = 1000
    best_sp=10000
    best_eo=10000



    sim_loss = 0
    for epoch in range(epochs):

        model.train()
        optimizer.zero_grad()
        embed, output  = model.forward(features, edge_index)
        # if name == 'bail' or name == 'occupation' or name =='sport' or name =='credit' or name == 'pokec_z':
        ff=features
        # else:
        #     ff=model.pre(features)
        # print(related_weights)
        # ff= (ff.squeeze() > 0).type_as(labels)
        # Binary Cross-Entropy
        preds = (output.squeeze() > 0).type_as(labels)
        loss_pred = criterion(output[data.idx_train], data.labels[data.idx_train].unsqueeze(1).float())
        # lss=criterion(l[data.idx_train], data.labels[data.idx_train].unsqueeze(1).float())
        # Fairness loss

        l1 = 0
        for related_attr, related_weight in zip(related_attrs, related_weights):
            l1 = 0
            indices = find_counterfactuals(embed[idx_complete_train], preds[idx_complete_train],
                                           ff[:, related_attr][idx_complete_train],
                                           indices_num)

            for i in range(indices_num):
                l1 += (1 - F.cosine_similarity(embed[idx_complete_train], embed[indices[:, i], :]).mean())
                # l1 += model.D(z1[idx_train], z2[idx_train][indices[:,i]]) / 2
                # l1 += torch.mean(torch.abs(embed[idx_complete_train] - embed[idx_complete_train][indices[:, i], :])**2)
                # l1 += torch.mean(
                #     torch.abs(embed[idx_complete_train] - embed[idx_complete_train][indices[:, i], :])) + torch.mean(
                #     (embed[idx_complete_train] - embed[idx_complete_train][indices[:, i], :]) ** 2)d
            sim_loss += weightSum * (l1) * related_weight

        loss_train = loss_pred + sim_loss

        auc_roc_train = roc_auc_score(labels.cpu().numpy()[idx_train], output.detach().cpu().numpy()[idx_train])
        loss_train.backward()
        optimizer.step()
        related_attrss = related_attrs
        with torch.no_grad():
            cor_losses = []
            for related_attrin in related_attrs:
                l1 = 0
                indices = find_counterfactuals(embed[idx_complete_train], preds[idx_complete_train],
                                               ff[:, related_attrin][idx_complete_train],
                                               indices_num)
                for i in range(indices_num):
                    l1 += 1 - F.cosine_similarity(embed[idx_complete_train], embed[indices[:, i], :]).mean()
                    # l1 += model.D(z1[idx_train], z2[idx_train][indices[:,i]]) / 2
                    # l1 += torch.mean(torch.abs(embed[idx_complete_train] - embed[idx_complete_train][indices[:, i], :])**2)
                    # # l1 += torch.mean(
                    #     torch.abs(embed[idx_complete_train] - embed[idx_complete_train][indices[:, i], :])) + torch.mean(
                    #     (embed[idx_complete_train] - embed[idx_complete_train][indices[:, i], :]) ** 2)
                cor_losses.append((l1).item())

            cor_losses = np.array(cor_losses)
            cor_order = np.argsort(cor_losses)
            beta = 0.00001
            v = cor_losses[cor_order[0]] + 2 * beta
            cor_sum = cor_losses[cor_order[0]]
            l = 1
            for i in range(cor_order.shape[0] - 1):
                if cor_losses[cor_order[i + 1]] < v:
                    cor_sum = cor_sum + cor_losses[cor_order[i + 1]]
                    v = (cor_sum + 2 * beta) / (i + 2)
                    l = l + 1
                else:
                    break

            # compute lambda
            for i in range(cor_order.shape[0]):
                if i < l:
                    related_weights[cor_order[i]] = (v - cor_losses[cor_order[i]]) / (2 * beta)
                else:
                    related_weights[cor_order[i]] = 0
        model.eval()

        embed, output = model(features, edge_index)

        # Binary Cross-Entropy
        loss_val_pred = criterion(output[data.idx_val], data.labels[data.idx_val].unsqueeze(1).float())
        # if name == 'bail' or name == 'occupation' or name =='sport' or name =='credit' or name == 'pokec_z' :
        ff = features
        # else:
        #     ff = model.pre(features)
        # ff = model.pre(features)
        # lss = criterion(l[data.idx_val], data.labels[data.idx_val].unsqueeze(1).float())

        sim_loss = 0
        for related_attr, related_weight in zip(related_attrs, related_attrss):
            indices = find_counterfactuals(embed[idx_complete_val], preds[idx_complete_val],
                                           ff[:, related_attr][idx_complete_val],
                                           indices_num)
            l1 = 0
            for i in range(indices_num):
                l1 += 1 - F.cosine_similarity(embed[idx_complete_val], embed[idx_complete_val][indices[:, i], :]).mean()
                # l1 += model.D(z1[idx_train], z2[idx_train][indices[:,i]]) / 2
                # l1+=torch.mean(torch.abs(embed[idx_complete_val]- embed[idx_complete_val][indices[:, i],:])**2)
                # l1 += torch.mean(torch.abs(embed[idx_complete_val] - embed[idx_complete_val][indices[:, i],:])) + torch.mean(
                #     (embed[idx_complete_val] - embed[idx_complete_val][indices[:, i],:]) ** 2)
            sim_loss += (l1) * weightSum * related_weight
        # Total loss
        loss_val = loss_val_pred + sim_loss
        output_preds = (output.squeeze() > 0).type_as(data.labels)
        acc_val = accuracy_score(data.labels[data.idx_val].cpu().numpy(), output_preds[data.idx_val].cpu().numpy())
        auc_roc_val = roc_auc_score(labels.cpu().numpy()[idx_val], output.detach().cpu().numpy()[idx_val])




        if epoch == 0:
            best_val = auc_roc_val

        preds = (output.squeeze() > 0).type_as(labels)
        f1_val = f1_score(labels[idx_val].cpu().numpy(), preds[idx_val].cpu().numpy())
        parity_val, equality_val = fair_metric(preds[idx_val].detach().cpu().numpy(), labels[idx_val].cpu().numpy(),
                                               sens[idx_val].numpy())



        acc_test = accuracy_score(data.labels[data.idx_test].cpu().numpy(), preds[data.idx_test].cpu().numpy())
        auc_roc_test = roc_auc_score(labels.cpu().numpy()[idx_test], output.detach().cpu().numpy()[idx_test])



        # if epoch%1 == 0:
        best_val = auc_roc_val
        preds = (output.squeeze() > 0).type_as(labels)
        f1_test = f1_score(labels[idx_test].cpu().numpy(), preds[idx_test].cpu().numpy())
        parity_test, equality_test = fair_metric(preds[idx_test].detach().cpu().numpy(), labels[idx_test].cpu().numpy(),
                                               sens[idx_test].numpy())




        print(
            f"[Train] Epoch {epoch}:train_loss: {loss_train.item():.4f} | train_auc_roc: {auc_roc_train:.4f} | val_loss: {loss_val.item():.4f} | val_auc_roc: {auc_roc_val:.4f}")
        print(
            f"[Val] Epoch {epoch}: acc: {acc_val:.4f} | f1: {f1_val:.4f} | parity: {parity_val:.4f} | equality: {equality_val:.4f}")
        print(
            f"[test] Epoch {epoch}: acc: {acc_test:.4f} | f1: {f1_test:.4f} | parity: {parity_test:.4f} | equality: {equality_test:.4f}")

        # if acc_val < 0.65 :
        #
        #     break

        # if (parity_val + equality_val) < best_sp :
        #     best_sp = (parity_val + equality_val)
        #     # torch.save(model.state_dict(), save_name)
        # elif (epoch ==0):
        #     print()
        # else:
        #     break
        # else:
        #     break

        # if epoch >5:
        #     if epoch % 1 == 0:
        #
        #pockec_n_GCN

        # if name=='credit':
        #
        #     if acc_val >0.70:
        #         if (parity_val + equality_val) < best_sp:
        #             best_sp = (parity_val + equality_val)
        #             torch.save(model.state_dict(), save_name)
        #     else:
        #         break
        #
        #
        #
        #
        # elif name=='nba'or name =='occupation':
        #
        #     if (parity_val + equality_val) < best_sp :
        #         best_sp = (parity_val + equality_val)
        #         torch.save(model.state_dict(), save_name)
        #     # else:
        #     #     break
        #
        # elif name =='german':
        #     if (parity_val + equality_val) < best_sp :
        #         best_sp = (parity_val + equality_val)
        #         torch.save(model.state_dict(), save_name)
        #     # else:
        #     #     break
        # elif name=='bail':
        #
        #             if loss_val.item() < best_loss and epoch >5:
        #             best_loss = loss_val.item()
        #         torch.save(model.state_dict(), save_name)
        # else:
        #     if model.name=='GCN' or model.name ==' SAGE' or model.name =='GAT':
        #         if (parity_val + equality_val) < best_sp :
        #             best_sp = (parity_val + equality_val)
        #             torch.save(model.state_dict(), save_name)
        #
        #         else:
        #             break
        #     else:
        if name =='bail':
            # if model.name == 'GIN':
            #     if acc_val < 0.75 :
            #         break
            #     else:
                    # if loss_val.item() < best_loss:
                    #     best_loss = loss_val.item()
                torch.save(model.state_dict(), save_name)
            # if model.name == 'GCN':
            #     if acc_val < 0.85 :
            #         break
            #     else:
                    # if loss_val.item() < best_loss:
                    #     best_loss = loss_val.item()
                torch.save(model.state_dict(), save_name)
        if name =='pokec_z':
            if model.name == 'GIN':
                if acc_val < 0.69 :
                    break
                else:
                    # if loss_val.item() < best_loss:
                    #     best_loss = loss_val.item()
                    torch.save(model.state_dict(), save_name)
            if model.name == 'GCN':
                if acc_val < 0.65 :
                    break
                else:
                    # if loss_val.item() < best_loss:
                    #     best_loss = loss_val.item()
                    torch.save(model.state_dict(), save_name)
        if name =='pokec_n':
            if model.name == 'GIN':
                if acc_val < 0.66 :
                    break
                else:
                    # if loss_val.item() < best_loss:
                    #     best_loss = loss_val.item()
                    torch.save(model.state_dict(), save_name)
            if model.name == 'GCN':
                if acc_val < 0.65 :
                    break
                else:
                    # if loss_val.item() < best_loss:
                    #     best_loss = loss_val.item()
                    torch.save(model.state_dict(), save_name)
        if name == 'occupation':
            if model.name == 'GIN':
                if acc_val < 0.81:
                    break
                else:
                    # if loss_val.item() < best_loss:
                    #     best_loss = loss_val.item()
                    torch.save(model.state_dict(), save_name)
            if model.name == 'GCN':
                if acc_val < 0.65:
                    break
                else:
                    # if loss_val.item() < best_loss:
                    #     best_loss = loss_val.item()
                    torch.save(model.state_dict(), save_name)
        if name == 'nba':
            if model.name == 'GIN':
                # if acc_val < 0.60:
                #     break
                # else:
                #     if loss_val.item() < best_loss:
                #         best_loss = loss_val.item()
                torch.save(model.state_dict(), save_name)
            if model.name == 'GCN':
                # if acc_val < 0.65:
                #     break
                # else:
                #     # if loss_val.item() < best_loss:
                #     #     best_loss = loss_val.item()
                torch.save(model.state_dict(), save_name)
        if name == 'credit':
            if model.name == 'GIN':
                if acc_val < 0.73:
                    break
                else:
                    # if loss_val.item() < best_loss:
                    #     best_loss = loss_val.item()
                    torch.save(model.state_dict(), save_name)
            if model.name == 'GCN':
                if acc_val < 0.73:
                    break
                else:
                    # if loss_val.item() < best_loss:
                    #     best_loss = loss_val.item()
                    torch.save(model.state_dict(), save_name)

        # torch.save(model.state_dict(), save_name)
def train_vanilla(model, optimizer, criterion, epochs, data, save_name):
    best_loss = 100
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        h, output = model(data.features, data.edge_index)
        loss_train = criterion(output[data.idx_train], data.labels[data.idx_train].unsqueeze(1).float())
        loss_train.backward()
        optimizer.step()

        model.eval()
        h, output = model(data.features, data.edge_index)
        loss_val = criterion(output[data.idx_val], data.labels[data.idx_val].unsqueeze(1).float())

        if loss_val.item() < best_loss:
            best_loss = loss_val.item()
            torch.save(model.state_dict(), save_name)
from sklearn.cluster import KMeans
import numbers
def groupTPR(p_predict, y_true, group_label, ind):
    p_predict=p_predict.detach().cpu().numpy()
    group_set = set(group_label)
    if len(group_set) > 1 and isinstance(list(group_set)[0], numbers.Number):
        kmeans = KMeans(n_clusters=2, random_state=0).fit(group_label.reshape(-1,1))
        group_label = group_label
        group_label = kmeans.predict(group_label.reshape(-1,1))
        group_set = set(group_label.cpu())
    else:
        group_label = group_label
        group_set = set(group_label.cpu())

    group_tpr = []
    y_true=y_true
    group_label=group_label
    for group in group_set:
        group_true_ind=(y_true == 1) & (group_label == group)
        cur_tpr = p_predict[group_true_ind.cpu(),:].mean()
        if not np.isnan(cur_tpr):
            group_tpr.append(cur_tpr)
    return group_tpr

def groupTNR(p_predict, y_true, group_label, ind):
    group_set = set(group_label)
    if len(group_set) > 5 and isinstance(list(group_set)[0], numbers.Number):
        kmeans = KMeans(n_clusters=5, random_state=0).fit(group_label.reshape(-1,1))
        group_label = group_label[ind.int()]
        group_label = kmeans.predict(group_label.reshape(-1,1))
        group_set = set(group_label)
    else:
        group_label = group_label[ind.int()]
        group_set = set(group_label)

    group_fnr = []
    for group in group_set:
        group_true_ind = np.array([ a==0 and b ==group for a, b in zip(y_true,group_label)])
        cur_fnr = p_predict[group_true_ind,:][:,0].mean()
        if not cur_fnr.isnan():
            group_fnr.append(cur_fnr)

    return group_fnr


def train_RIF(model, optimizer, criterion, epochs, data, save_name):
    best_loss = 100
    num = 1
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        h, output = model(data.features, data.edge_index)
        loss_train = criterion(output[data.idx_train], data.labels[data.idx_train].unsqueeze(1).float())
        all_indices = list(range(num))
        related_weights = []
        related_attrs = all_indices



def train_student(model, optimizer, criterion_bce, criterion_kd, args, data, save_name, soft_target):
    best_loss = 100
    best_result = 0
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        h, output = model(data.features, data.edge_index)

        loss_bce_train = criterion_bce(output[data.idx_train], data.labels[data.idx_train].unsqueeze(1).float())
        loss_kd_train = criterion_kd(h[data.idx_train], soft_target[data.idx_train])
        if epoch == 0:
            weight_compute = AdaWeight(loss_bce_train, loss_kd_train, lr=args.lr_w, gamma=args.gamma)
        lad1, lad2 = weight_compute.compute(loss_bce_train.item(), loss_kd_train.item())
        loss_train = lad1 * loss_bce_train + lad2 * loss_kd_train

        loss_train.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            h, output = model(data.features, data.edge_index)
        loss_bce_val = criterion_bce(output[data.idx_val], data.labels[data.idx_val].unsqueeze(1).float())
        loss_kd_val = criterion_kd(h[data.idx_val], soft_target[data.idx_val])
        loss_val = loss_bce_val + loss_kd_val
        output_preds = (output.squeeze() > 0).type_as(data.labels)
        acc = accuracy_score(data.labels[data.idx_val].cpu().numpy(), output_preds[data.idx_val].cpu().numpy())
        parity, equality = fair_metric(output_preds[data.idx_val].cpu().numpy(),
                                       data.labels[data.idx_val].cpu().numpy(),
                                       data.sens[data.idx_val].numpy())

        if args.dataset == 'pokec_z' and args.model == 'gin':
            if acc - 3*(parity + equality) > best_result:
                best_result = acc - 3*(parity + equality)
                torch.save(model.state_dict(), save_name)
        else:
            if loss_val.item() < best_loss:
                best_loss = loss_val.item()
                torch.save(model.state_dict(), save_name)
                # print(f"[Train] Epoch {epoch}:bce_loss: {loss_bce_train.item():.4f} | kd_loss: {loss_kd_train.item():.4f} "
                #       f"| total_loss: {loss_train.item():.4f} | lad1: {lad1:.4f}, lad2: {lad2:.4f}")



def train_k(model, optimizer, criterion, epochs, data, save_name):
    all_indices = list(range(3))
    related_weights = []
    related_attrs = all_indices

    for zz in range(3):
        related_weights.append(0.1)
    for i in range(15):
        optimizer.zero_grad()
        h, output = model(data.features, data.edge_index)
        loss_train = criterion(output[data.idx_train], data.labels[data.idx_train].unsqueeze(1).float())
        preds = (output.squeeze() > 0).type_as(data.labels)
        loss = loss_train

        for related_attr, related_weight in zip(related_attrs, related_weights):
            group_TPR = groupTPR(output[data.idx_train], data.labels[data.idx_train], data.features[data.idx_train,related_attr], data.idx_train)
            # group_TPR_loss = (group_TPR - (sum(group_TPR)/len(group_TPR)).detach()).sum()*related_weight
            group_TPR_loss = torch.square(torch.tensor(max(group_TPR)- min(group_TPR), dtype=torch.float32))

            # group_TNR = utils.groupTNR(p_y, y, np.array(data_frame[related_attr].tolist()), ind)
            # group_TNR_loss = torch.square(max(group_TNR).detach() - min(group_TNR))
            # print('classification loss: {}, group TPR loss: {}, group TNR loss: {}'.format(loss.item(), group_TPR_loss.item(), group_TNR_loss.item()))
            # print('classification loss: {}, group TPR loss: {}'.format(loss.item(), group_TPR_loss.item()))
            loss = loss + group_TPR_loss * related_weight
        loss.backward()
        optimizer.step()
    torch.save(model.state_dict(), save_name)

def train_KSMOTE(model, optimizer, criterion, epochs, data, save_name):
    best_loss = 100
    num=3

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        h, output = model(data.features, data.edge_index)
        loss_train = criterion(output[data.idx_train], data.labels[data.idx_train].unsqueeze(1).float())
        all_columns = list(range(data.features.shape[1]))
        random_columns = random.sample(all_columns, 3)
        # all_indices = list(range(num))
        related_weights = []
        related_attrs = random_columns
        for zz in range(num):
            related_weights.append(0.1)
        UPDATE_MODEL_ITERS = 1
        UPDATE_WEIGHT_ITERS = 1
        # print(related_weights)
        # update model



        for related_attr, related_weight in zip(related_attrs, related_weights):

            cor_loss = torch.sum(torch.abs(torch.mean(torch.mul(
                data.features[:, related_attr].reshape(1, data.features.shape[0], -1) - data.features[:, related_attr].mean(dim=0).reshape(1, 1,
                                                                                                             -1),
                (output - output.mean(dim=0)).transpose(0, 1).reshape((-1, output.shape[0], 1))), dim=1)))

            # print('classification loss: {}, feature correlation loss: {}'.format(loss.item(), cor_loss.item()))
            loss = loss_train + cor_loss * related_weight * 0.3

        loss.backward()
        optimizer.step()

        # update weights
        # ipdb.set_trace()
        for iter in range(UPDATE_WEIGHT_ITERS):
            with torch.no_grad():
                h, output = model(data.features, data.edge_index)
                loss_train = criterion(output[data.idx_train], data.labels[data.idx_train].unsqueeze(1).float())

                cor_losses = []
                for related_attr in related_attrs:

                    cor_loss = torch.sum(torch.abs(torch.mean(torch.mul(
                        data.features[:, related_attr].reshape(1, data.features.shape[0], -1) - data.features[:, related_attr].mean(dim=0).reshape(1,
                                                                                                                     1,
                                                                                                                     -1),
                        (output - output.mean(dim=0)).transpose(0, 1).reshape((-1, output.shape[0], 1))), dim=1)))

                    cor_losses.append(cor_loss.item())

                cor_losses = np.array(cor_losses)

                cor_order = np.argsort(cor_losses)

                # compute -v. represent it as v.
                beta = 0.0002
                v = cor_losses[cor_order[0]] + 2 * beta
                cor_sum = cor_losses[cor_order[0]]
                l = 1
                for i in range(cor_order.shape[0] - 1):
                    if cor_losses[cor_order[i + 1]] < v:
                        cor_sum = cor_sum + cor_losses[cor_order[i + 1]]
                        v = (cor_sum + 2 * beta) / (i + 2)
                        l = l + 1
                    else:
                        break

                # compute lambda
                for i in range(cor_order.shape[0]):
                    if i < l:
                        related_weights[cor_order[i]] = (v - cor_losses[cor_order[i]]) / (2 * beta)
                    else:
                        related_weights[cor_order[i]] = 0

                # print(1)

    torch.save(model.state_dict(), save_name)


def load_credit(dataset, sens_attr="Age", predict_attr="NoDefaultNextMonth", path="./dataset/credit/", label_number=1000):
    # print('Loading {} dataset from {}'.format(dataset, path))
    idx_features_labels = pd.read_csv("/root/autodl-tmp/project/FairGKD-master/datasets/credit/credit.csv")
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)
    header.remove('Single')
    rela = []  # 假设你要与第2、3、4列属性计算相关性
    for i in range(11):
        rela.append(header[i])

    # for s in rela:
    #     # 调用函数并传递参数
    #     correlation_result = cal_correlation(idx_features_labels, sens_attr, s)
    #
    #     # 打印结果
    #     print(s, "相关性总和:", correlation_result)

#    # Normalize MaxBillAmountOverLast6Months
#    idx_features_labels['MaxBillAmountOverLast6Months'] = (idx_features_labels['MaxBillAmountOverLast6Months']-idx_features_labels['MaxBillAmountOverLast6Months'].mean())/idx_features_labels['MaxBillAmountOverLast6Months'].std()
#
#    # Normalize MaxPaymentAmountOverLast6Months
#    idx_features_labels['MaxPaymentAmountOverLast6Months'] = (idx_features_labels['MaxPaymentAmountOverLast6Months'] - idx_features_labels['MaxPaymentAmountOverLast6Months'].mean())/idx_features_labels['MaxPaymentAmountOverLast6Months'].std()
#
#    # Normalize MostRecentBillAmount
#    idx_features_labels['MostRecentBillAmount'] = (idx_features_labels['MostRecentBillAmount']-idx_features_labels['MostRecentBillAmount'].mean())/idx_features_labels['MostRecentBillAmount'].std()
#
#    # Normalize MostRecentPaymentAmount
#    idx_features_labels['MostRecentPaymentAmount'] = (idx_features_labels['MostRecentPaymentAmount']-idx_features_labels['MostRecentPaymentAmount'].mean())/idx_features_labels['MostRecentPaymentAmount'].std()
#
#    # Normalize TotalMonthsOverdue
#    idx_features_labels['TotalMonthsOverdue'] = (idx_features_labels['TotalMonthsOverdue']-idx_features_labels['TotalMonthsOverdue'].mean())/idx_features_labels['TotalMonthsOverdue'].std()

    # build relationship
    if os.path.exists(f'/root/autodl-tmp/project/FairGKD-master/datasets/credit/credit_edges.txt'):
        edges_unordered = np.genfromtxt(f'/root/autodl-tmp/project/FairGKD-master/datasets/credit/credit_edges.txt').astype('int')
    else:
        edges_unordered = build_relationship(idx_features_labels[header], thresh=0.7)
        np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values
    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    import random
    random.seed(20)
    label_idx_0 = np.where(labels==0)[0]
    label_idx_1 = np.where(labels==1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)

    idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number//2)], label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number//2)])
    idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))], label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])

    sens = idx_features_labels[sens_attr].values.astype(int)
    sens = torch.FloatTensor(sens)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    # features = torch.cat((features[:, :1], features[:, 2:]), dim=1)
    return adj, features, labels, idx_train, idx_val, idx_test, sens

def load_german(dataset, sens_attr="Gender", predict_attr="GoodCustomer", path="../dataset/german/", label_number=1000):
    # print('Loading {} dataset from {}'.format(dataset, path))
    idx_features_labels = pd.read_csv('/root/autodl-tmp/project/FairGKD-master/datasets/german/german.csv')
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)
    header.remove('OtherLoansAtStore')
    header.remove('PurposeOfLoan')
    # min_values = features.min(axis=0)[0]
    # max_values = features.max(axis=0)[0]
    # return 2 * (features - min_values).div(max_values - min_values) - 1
    rela = []  # 假设你要与第2、3、4列属性计算相关性
    # for i in range(25):
    #     rela.append(header[i])

    # for s in rela:
    #     # 调用函数并传递参数
    #     correlation_result = cal_correlation(idx_features_labels, sens_attr, s)
    #
    #     # 打印结果
    #     print(s, "相关性总和:", correlation_result)
    # Sensitive Attribute
    idx_features_labels['Gender'][idx_features_labels['Gender'] == 'Female'] = 1
    idx_features_labels['Gender'][idx_features_labels['Gender'] == 'Male'] = 0

#    for i in range(idx_features_labels['PurposeOfLoan'].unique().shape[0]):
#        val = idx_features_labels['PurposeOfLoan'].unique()[i]
#        idx_features_labels['PurposeOfLoan'][idx_features_labels['PurposeOfLoan'] == val] = i

#    # Normalize LoanAmount
#     idx_features_labels['LoanAmount'] = 2*(idx_features_labels['LoanAmount']-idx_features_labels['LoanAmount'].min()).div(idx_features_labels['LoanAmount'].max() - idx_features_labels['LoanAmount'].min()) - 1
#
# #    # Normalize Age
#     idx_features_labels['Age'] = 2*(idx_features_labels['Age']-idx_features_labels['Age'].min()).div(idx_features_labels['Age'].max() - idx_features_labels['Age'].min()) - 1
#
# #    # Normalize LoanDuration
#     idx_features_labels['LoanDuration'] = 2*(idx_features_labels['LoanDuration']-idx_features_labels['LoanDuration'].min()).div(idx_features_labels['LoanDuration'].max() - idx_features_labels['LoanDuration'].min()) - 1

    # build relationship
    if os.path.exists(f'/root/autodl-tmp/project/FairGKD-master/datasets/german/german_edges.txt'):
        edges_unordered = np.genfromtxt(f'/root/autodl-tmp/project/FairGKD-master/datasets/german/german_edges.txt').astype('int')
    else:
        edges_unordered = build_relationship(idx_features_labels[header], thresh=0.8)
        np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values
    labels[labels == -1] = 0

    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    import random
    random.seed(20)
    label_idx_0 = np.where(labels==0)[0]
    label_idx_1 = np.where(labels==1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)

    idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number//2)], label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number//2)])
    idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))], label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])

    sens = idx_features_labels[sens_attr].values.astype(int)
    sens = torch.FloatTensor(sens)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)


    # 假设你的张量 X_raw 是一个 NumPy 数组或张量对象
    # sens_attr 和 related_attr 是列的序号
    # 创建 Pandas DataFrame 对象


    # 打印相关性结果

    # features = features[:, 1:]
    return adj, features, labels, idx_train, idx_val, idx_test, sens

def evaluation(model, weight_path, data):
    model.load_state_dict(torch.load(weight_path))
    model.eval()

    with torch.no_grad():
        h, output = model(data.features, data.edge_index)
    output_preds = (output.squeeze() > 0).type_as(data.labels)
    auc_test = roc_auc_score(data.labels.cpu().numpy()[data.idx_test.cpu()],
                                 output.detach().cpu().numpy()[data.idx_test.cpu()])
    f1_test = f1_score(data.labels[data.idx_test].cpu().numpy(), output_preds[data.idx_test].cpu().numpy())
    acc_test = accuracy_score(data.labels[data.idx_test].cpu().numpy(), output_preds[data.idx_test].cpu().numpy())
    dp_test, eo_test = fair_metric(output_preds[data.idx_test].cpu().numpy(), data.labels[data.idx_test].cpu().numpy(),
                                     data.sens[data.idx_test].numpy())
    return auc_test, f1_test, acc_test, dp_test, eo_test


def fair_metric(pred, labels, sens):
    idx_s0 = sens==0
    idx_s1 = sens==1
    idx_s0_y1 = np.bitwise_and(idx_s0, labels==1)
    idx_s1_y1 = np.bitwise_and(idx_s1, labels==1)
    parity = abs(sum(pred[idx_s0])/sum(idx_s0)-sum(pred[idx_s1])/sum(idx_s1))
    equality = abs(sum(pred[idx_s0_y1])/sum(idx_s0_y1)-sum(pred[idx_s1_y1])/sum(idx_s1_y1))
    return parity.item(), equality.item()



class AdaWeight:
    def __init__(self, loss1_init, loss2_init, weight_loss1=0.5, lr=0.025, gamma=0.25):
        self.loss1_init = loss1_init
        self.loss2_init = loss2_init
        self.weight_loss1 = weight_loss1
        self.lr = lr
        self.gamma = gamma

    def compute(self, loss1, loss2):
        rela_loss1 = (loss1 / self.loss1_init.item())**self.gamma
        rela_loss2 = (loss2 / self.loss2_init.item())**self.gamma
        rela_weight_loss1 = rela_loss1 / (rela_loss1 + rela_loss2)
        self.weight_loss1 = self.lr * rela_weight_loss1 + (1 - self.lr) * self.weight_loss1
        self.weight_loss2 = 1 - self.weight_loss1
        return self.weight_loss1, self.weight_loss2


class ContLoss(_Loss):
    def __init__(self, reduction='mean', tem: float=0.5):
        super(ContLoss, self).__init__()
        self.reduction = reduction
        self.tem: float = tem

    def sim(self, h1: torch.Tensor, h2: torch.Tensor):
        h1 = F.normalize(h1)
        h2 = F.normalize(h2)
        return torch.mm(h1, h2.t())

    def loss(self, h1: torch.Tensor, h2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tem)
        intra_sim = f(self.sim(h1, h1))
        inter_sim = f(self.sim(h1, h2))
        return -torch.log(inter_sim.diag() / (inter_sim.sum(1) + intra_sim.sum(1) - intra_sim.diag()))

    def forward(self, h1: torch.Tensor, h2: torch.Tensor):
        l1 = self.loss(h1, h2)
        l2 = self.loss(h2, h1)
        ret = (l1 + l2) / 0.5
        ret = ret.mean() if self.reduction=='mean' else ret.sum()
        return ret
