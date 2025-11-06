import sys
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch_sparse
import pickle as pkl
import networkx as nx
from time import perf_counter
from torch_geometric.utils import subgraph, from_scipy_sparse_matrix
from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.data import DataLoader
from ogb.nodeproppred import PygNodePropPredDataset

from sklearn import metrics   


class Chains(InMemoryDataset):
    def __init__(self, root: str, name: str, target='status', transform=None, pre_transform=None):
        self.name = name
        self.target = target
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self.name, 'raw')

    @property
    def raw_file_names(self) -> str:
        return self.name + '.mat'

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self.name, 'processed')

    @property
    def processed_file_names(self) -> str:
        return f'data-{self.name}-{self.target}.pt'

    def download(self):
        pass

    def parse_index_file(filename):
        """Parse index file."""
        index = []
        for line in open(filename):
            index.append(int(line.strip()))
        return index
    

    def fetch_normalization(self, type):
        switcher = {
            'AugNormAdj': self.aug_normalized_adjacency,  # A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
        }
        func = switcher.get(type, lambda: "Invalid normalization technique.")
        return func


    def row_normalize(self, mx):
        """Row-normalize sparse matrix"""
        pass
        '''
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        '''
        return mx
    

    def preprocess_citation(self, adj, features, normalization="FirstOrderGCN"):
        adj_normalizer = self.fetch_normalization(normalization)
        adj = adj_normalizer(adj)
        features = self.row_normalize(features)
        return adj, features
    
    def load_citation_chain(self, normalization, device, need_orig=False):
        """load the synthetic dataset: chain"""
        r = np.random.RandomState(42)

        test_chains = self.name # 'chain-1' or 'chain-2' or 'chain-3'      

        if test_chains == 'chain-1':
            c = 2 # num of classes
            n = 3 # chains for each class
            l = 8 # length of chain
            f = 5 # feature dimension
            tn = 8  # train nodes
            vl = 8 # val nodes
            tt = 32 # test nodes
            noise = 0.00

        if test_chains == 'chain-2':
            c = 2
            n = 3
            l = 10
            f = 5
            tn = 10
            vl = 10
            tt = 40
            noise = 0.00

        if test_chains == 'chain-3':
            c = 2
            n = 3
            l = 15
            f = 5
            tn = 15
            vl = 15
            tt = 60
            noise = 0.00

        if test_chains == 'chain-4':
            c = 2
            n = 5
            l = 15
            f = 5
            tn = 25
            vl = 25
            tt = 100
            noise = 0.00

        if test_chains == 'chain-5':
            c = 2
            n = 50
            l = 15
            f = 5
            tn = 250
            vl = 250
            tt = 1000
            noise = 0.00

        chain_adj = sp.coo_matrix((np.ones(l-1), (np.arange(l-1), np.arange(1, l))), shape=(l, l))
        adj = sp.block_diag([chain_adj for _ in range(c*n)]) # square matrix N = c*n*l

        features = r.uniform(-noise, noise, size=(c, n, l, f))
        #features = np.zeros_like(features)
        features[:, :, 0, :c] += np.eye(c).reshape(c, 1, c) # add class info to the first node of chains.
        features = features.reshape(-1, f)

        labels = np.eye(c).reshape(c, 1, 1, c).repeat(n, axis=1).repeat(l, axis=2) # one-hot labels
        labels = labels.reshape(-1, c)

        idx_random = np.arange(c*n*l)
        r.shuffle(idx_random)
        idx_train = idx_random[:tn]
        idx_val = idx_random[tn:tn+vl]
        idx_test = idx_random[tn+vl:tn+vl+tt]

        if need_orig:
            adj_orig = self.aug_normalized_adjacency(adj, need_orig=False)
            adj_orig = self.sparse_mx_to_torch_sparse_tensor(adj_orig).float()
            if device:
                adj_orig = adj_orig.to(device)

        adj, features = self.preprocess_citation(adj, features, normalization)

        #features = torch.FloatTensor(features).to(device)
        #labels = torch.LongTensor(labels).to(device)
        # porting to pytorch
        #features = torch.FloatTensor(np.array(features.todense() if sp.issparse(features) else features)).float().to(device)
        labels = torch.LongTensor(labels)
        labels = torch.max(labels, dim=1)[1]
        # 
        adj = self.sparse_mx_to_torch_sparse_tensor(adj, device=self.device).float()
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        # edge_index = torch.tensor(np.vstack((adj.row, adj.col)), dtype=torch.long)
        #x = torch.tensor(features, dtype=torch.float)
        # y = torch.tensor(labels.argmax(axis=1), dtype=torch.long)  # Convert one-hot to class indices

        
        features = torch.tensor(features, dtype=torch.float).to(device)
        adj = adj.to(device).coalesce() 
        labels = labels.to(device)
        idx_train = idx_train.to(device)
        idx_val = idx_val.to(device)
        idx_test = idx_test.to(device)
        
        edge_index = adj.coalesce().indices().to(device)
        #print("debug: labels", labels)
        data = Data(x=features, edge_index=edge_index, y=labels, train_mask=[], val_mask=[], test_mask=[])

        data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        for i in idx_train:
            data.train_mask[i] = True
        data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.val_mask[idx_val] = True
        data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.test_mask[idx_test] = True

        
        #data.__num_nodes__ = self.data.x.size(0)
        #data.y = data.y.view(-1)
        # return [adj, adj_orig] if need_orig else features, labels, idx_train, idx_val, idx_test, adj
        
        return data

    def process(self):
        data = self.load_citation_chain(normalization="AugNormAdj", device=self.device)
        data, slices = self.collate([data])
        torch.save((data,slices), self.processed_paths[0])

    def len(self):
        return 1  # Since we have one large graph

    def get(self, idx):
        data = torch.load(self.processed_paths[0])[0]
        return data
    
    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx, device=None):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        #print("debug: sparse_mx", sparse_mx)
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        tensor = torch.sparse.FloatTensor(indices, values, shape)
        if device is not None:
            tensor = tensor.to(device)
        return tensor


    def get_spectral_rad(sparse_tensor, tol=1e-5):
        """Compute spectral radius from a tensor"""
        A = sparse_tensor.data.coalesce().cpu()
        A_scipy = sp.coo_matrix((np.abs(A.values().numpy()), A.indices().numpy()), shape=A.shape)
        return np.abs(sp.linalg.eigs(A_scipy, k=1, return_eigenvectors=False)[0]) + tol

    def projection_norm_inf(A, kappa=0.99, transpose=False):
        """ project onto ||A||_inf <= kappa return updated A"""
        # TODO: speed up if needed
        v = kappa
        if transpose:
            A_np = A.T.clone().detach().cpu().numpy()
        else:
            A_np = A.clone().detach().cpu().numpy()
        x = np.abs(A_np).sum(axis=-1)
        for idx in np.where(x > v)[0]:
            # read the vector
            a_orig = A_np[idx, :]
            a_sign = np.sign(a_orig)
            a_abs = np.abs(a_orig)
            a = np.sort(a_abs)

            s = np.sum(a) - v
            l = float(len(a))
            for i in range(len(a)):
                # proposal: alpha <= a[i]
                if s / l > a[i]:
                    s -= a[i]
                    l -= 1
                else:
                    break
            alpha = s / l
            a = a_sign * np.maximum(a_abs - alpha, 0)
            # verify
            assert np.isclose(np.abs(a).sum(), v, atol=1e-4)
            # write back
            A_np[idx, :] = a
        A.data.copy_(torch.tensor(A_np.T if transpose else A_np, dtype=A.dtype, device=A.device))
        return A

    def projection_norm_inf_and_1(A, kappa_inf=0.99, kappa_1=None, inf_first=True):
        """ project onto ||A||_inf <= kappa return updated A"""
        # TODO: speed up if needed
        v_inf = kappa_inf
        v_1 = kappa_inf if kappa_1 is None else kappa_1
        A_np = A.clone().detach().cpu().numpy()
        if inf_first:
            A_np = projection_inf_np(A_np, v_inf)
            A_np = projection_inf_np(A_np.T, v_1).T
        else:
            A_np = projection_inf_np(A_np.T, v_1).T
            A_np = projection_inf_np(A_np, v_inf)
        A.data.copy_(torch.tensor(A_np, dtype=A.dtype, device=A.device))
        return A

    def projection_inf_np(A_np, v):
        x = np.abs(A_np).sum(axis=-1)
        for idx in np.where(x > v)[0]:
            # read the vector
            a_orig = A_np[idx, :]
            a_sign = np.sign(a_orig)
            a_abs = np.abs(a_orig)
            a = np.sort(a_abs)

            s = np.sum(a) - v
            l = float(len(a))
            for i in range(len(a)):
                # proposal: alpha <= a[i]
                if s / l > a[i]:
                    s -= a[i]
                    l -= 1
                else:
                    break
            alpha = s / l
            a = a_sign * np.maximum(a_abs - alpha, 0)
            # verify
            assert np.isclose(np.abs(a).sum(), v, atol=1e-6)
            # write back
            A_np[idx, :] = a
        return A_np

    def load_raw_graph(dataset_str = "amazon-all"):
        txt_file = 'data/' + dataset_str + '/adj_list.txt'
        graph = {}
        with open(txt_file, 'r') as f:
            cur_idx = 0
            for row in f:
                row = row.strip().split()
                adjs = []
                for j in range(1, len(row)):
                    adjs.append(int(row[j]))
                graph[cur_idx] = adjs
                cur_idx += 1
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        normalization="AugNormAdj"
        adj_normalizer = fetch_normalization(normalization)
        adj = adj_normalizer(adj)
        adj = sparse_mx_to_torch_sparse_tensor(adj).float()
        return adj

    def load_txt_data(dataset_str = "amazon-all", portion = '0.06'):
        adj = load_raw_graph(dataset_str)
        idx_train = list(np.loadtxt('data/' + dataset_str + '/train_idx-' + str(portion) + '.txt', dtype=int))
        idx_val = list(np.loadtxt('data/' + dataset_str + '/test_idx.txt', dtype=int))
        idx_test = list(np.loadtxt('data/' + dataset_str + '/test_idx.txt', dtype=int))
        labels = np.loadtxt('data/' + dataset_str + '/label.txt')
        with open('data/' + dataset_str + '/meta.txt', 'r') as f:
            num_nodes, num_class = [int(w) for w in f.readline().strip().split()]

        features = sp.identity(num_nodes)
        
        # porting to pytorch
        features = sparse_mx_to_torch_sparse_tensor(features).float()
        labels = torch.FloatTensor(labels)
        #labels = torch.max(labels, dim=1)[1]
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        return adj, features, labels, idx_train, idx_val, idx_test, num_nodes, num_class

    def sgc_precompute(features, adj, degree):
        t = perf_counter()
        adj_index = adj.coalesce().indices()
        adj_value = adj.coalesce().values()
        features_index = features.coalesce().indices()
        features_value = features.coalesce().values()
        m = adj.shape[0]
        n = adj.shape[1]
        k = features.shape[1]

        for i in range(degree):
            #features = torch.spmm(adj, features)
            features_index, features_value = torch_sparse.spspmm(adj_index, adj_value, features_index, features_value, m, n, k)
        precompute_time = perf_counter()-t
        return torch.sparse.FloatTensor(features_index, features_value, torch.Size(features.shape)), precompute_time

    def aug_normalized_adjacency(self, adj, need_orig=False):
        if not need_orig:
            adj = adj + sp.eye(adj.shape[0])
        adj = sp.coo_matrix(adj)
        row_sum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(row_sum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

   