import torch
import torch.nn.functional as F
from typing import Annotated, Optional
from torch import nn
from torch.optim import Adam, SGD, Optimizer
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from torch_geometric.transforms import ToSparseTensor
import torch_sparse
from torch_sparse import SparseTensor, matmul
from core import console
from core.args.utils import ArgInfo
from core.methods.node.base import NodeClassification
from core.models.multi_mlp import MultiMLP
from core.modules.base import Metrics
from core.modules.node.cm import ClassificationModule
from core.modules.node.em import EncoderModule

def get_normalized_adj(data, flag_norm: str = 'True', fill_degree=0):
    if flag_norm == 'False':
           # This is the original normalization method
           adj_norm = data.adj_t.to_dense()
           x_min_degree = adj_norm.sum(dim=0).min()
    elif flag_norm == 'True': 
        adj = data.adj_t
        
        # add self loops
        num_nodes = adj.size(0)

        adj = adj.to_symmetric()
        set_value = torch.ones(adj.nnz(), device=adj.device())
        adj = adj.set_value(set_value,layout='coo')   

        deg = torch_sparse.sum(adj, dim=1)
        deg = deg.to(adj.device())
        diag = torch.ones(num_nodes) * fill_degree
        diag = diag.to(adj.device())
        diag = torch.clamp(diag - deg, min=0)
 
        adj_t = torch_sparse.set_diag(adj, diag)
        deg = torch_sparse.sum(adj_t, dim=1)
        x_min_degree = deg.min() 
        # add self loops

        '''
        adj_dense = adj.to_dense()
        # fill_diag(100)
        deg = adj_dense.sum(dim=0)
        fill_matrix = torch.ones_like(deg) * fill_degree
        new_fill = fill_matrix - deg
        new_fill = torch.clamp(new_fill, min=0)
        # adj_dense.fill_diagonal_(fill_degree)  # TEST PARAMETER (min_degree)
        adj_dense = adj_dense + torch.diag(new_fill)

        x_min_degree = adj_dense.sum(dim=0).min()
        # print("debugging...min degree", x_min_degree, "debugging...")
        '''

        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = torch_sparse.mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = torch_sparse.mul(adj_t, deg_inv_sqrt.view(1, -1))
        # adj_norm = adj_norm.to_sparse()
    else: 
        raise Exception('Not supported normalization flag. Please specify either True or False.')
    return adj_t, x_min_degree


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, device, bound_lipschitz, bias=None):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features 
        self.device = device
        self.bound_lipschitz = bound_lipschitz 
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features)).to(self.device)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features)).to(self.device)
        else:
            self.register_parameter('bias', None)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # check if gpu is available
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input, adj):
        adj = adj.to(self.device, non_blocking=True)
        # support = input * self.bound_lipschitz  # weight 改成<1 @self.weight TEST PARAMETER
        support = input
        support = support.to(self.device, non_blocking=True)
        output = matmul(adj, support)
        if self.bias is not None:
            return output + self.bias.to(self.device, non_blocking=True)
        else:
            return output


class Caribou (NodeClassification):
    """Non-private Caribou method"""

    supported_activations = {
        'relu': torch.relu_,
        'selu': torch.selu_,
        'tanh': torch.tanh,
    }

    def __init__(self,
                 num_classes,
                 hops:            Annotated[int,   ArgInfo(help='number of hops', option='-k')] = 2,
                 hidden_dim:      Annotated[int,   ArgInfo(help='dimension of the hidden layers')] = 16,
                 encoder_layers:  Annotated[int,   ArgInfo(help='number of encoder MLP layers')] = 2,
                 base_layers:     Annotated[int,   ArgInfo(help='number of base MLP layers')] = 1,
                 head_layers:     Annotated[int,   ArgInfo(help='number of head MLP layers')] = 1,
                 combine:         Annotated[str,   ArgInfo(help='combination type of transformed hops', choices=MultiMLP.supported_combinations)] = 'cat',
                 activation:      Annotated[str,   ArgInfo(help='type of activation function', choices=supported_activations)] = 'selu',
                 dropout:         Annotated[float, ArgInfo(help='dropout rate')] = 0.0,
                 batch_norm:      Annotated[bool,  ArgInfo(help='if true, then model uses batch normalization')] = True,
                 encoder_epochs:  Annotated[int,   ArgInfo(help='number of epochs for encoder pre-training (ignored if encoder_layers=0)')] = 100,
                 weight_decay:    Annotated[float,   ArgInfo(help='weight decay for Caribou Adam')] = 0.01,
                 bound_lipschitz: Annotated[float,   ArgInfo(help='lipschitz constraint (0,1)')] = 0.9,
                 alpha_1:         Annotated[float,   ArgInfo(help='alpha_1 + alpha_2 = 1')] = 0.95,
                 beta:         Annotated[float,   ArgInfo(help='beta > 0')] = 0.9,
                 fill_degree:     Annotated[int,   ArgInfo(help='number of min_degree to be newly filled: the number of self loops')] = 0,
                 **kwargs:        Annotated[dict,  ArgInfo(help='extra options passed to base class', bases=[NodeClassification])]
                 ):

        super().__init__(num_classes, **kwargs)

        if encoder_layers == 0 and encoder_epochs > 0:
            console.warning('encoder_layers is 0, setting encoder_epochs to 0')
            encoder_epochs = 0

        
        self.hops = torch.tensor(hops).to(self.device) # number of graph convolutional layers
        # self.hops = hops
        self.fill_degree = fill_degree
        self.encoder_layers = encoder_layers
        self.encoder_epochs = encoder_epochs
        self.weight_decay = weight_decay
        self.bound_lipschitz = bound_lipschitz # parameter range (0,1) 
        self.alpha_1 = alpha_1
        self.alpha_2 = 1 - self.alpha_1
        self.beta = beta
        activation_fn = self.supported_activations[activation]
        # self.gcn_layer = GraphConvolution(hidden_dim,hidden_dim, bias=True)
        # initialize multiple GCN layers according to the number of hops
        self.gcn_layers = nn.ModuleList([GraphConvolution(hidden_dim, hidden_dim, self.device, self.bound_lipschitz) for _ in range(hops)])


        self._encoder = EncoderModule(
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            encoder_layers=encoder_layers,
            head_layers=1,
            normalize=True,
            activation_fn=activation_fn,
            dropout=dropout,
            batch_norm=batch_norm,
        )

        self._classifier = ClassificationModule(
            num_channels=hops+1,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            base_layers=base_layers,
            head_layers=head_layers,
            combination=combine,
            activation_fn=activation_fn,
            dropout=dropout,
            batch_norm=batch_norm,
        )

    @property
    def classifier(self) -> ClassificationModule:
        return self._classifier

    def reset_parameters(self):
        self._encoder.reset_parameters()
        super().reset_parameters()

    def fit(self, data: Data, learn: str = 'transductive', prefix: str = '') -> Metrics:
        self.learning_setting = learn
        
        if learn == 'transductive':
            
            self.data = data.clone().detach().to(self.device, non_blocking=True)

            # pre-train encoder
            if self.encoder_layers > 0:
                pass 
                #self.data = self.pretrain_encoder(self.data, prefix=prefix)

            feature_initial = self.data
            # compute aggregations
            self.data = self.compute_aggregations(self.data)

            print("debugging...self.data", type(self.data), "debugging...")
            print(f"debugging...data.x is_sparse: {data.x.is_sparse}")
            # train classifier
            return super().fit(self.data, learn=learn, prefix=prefix)
        
        elif learn == 'inductive':

            # we first record the clone of data in self.data
            self.data = data.clone().detach().to(self.device, non_blocking=True)
            
            # in inductive setting we only know the information about the nodes in training set
            # the input data contains the complete graph information so we need to first extract the training dataset
            data = data.to(self.device, non_blocking=True)

            # extract the training graph (train_mask + val_mask)
            node_indices = torch.LongTensor([i for i in range(data.x.size(0))]).to(self.device)
            train_node_indices = node_indices[data.train_mask | data.val_mask]
            row, col, _ = data.adj_t.t().coo()
            edge_index = torch.stack([row, col], dim=0)
            
            train_edge_index, _ = subgraph(train_node_indices, edge_index, relabel_nodes=False)
            data.edge_index = train_edge_index  # change the edge index
            data = ToSparseTensor()(data)  # now data.adj_t has changed
            
            # pre-train encoder
            if self.encoder_layers > 0:
                pass
                #data = self.pretrain_encoder(data, prefix=prefix)

            # compute aggregations
            data = self.compute_aggregations(data)

            # train classifier
            return super().fit(data, learn=learn, prefix=prefix)
        
        else:
            
            raise Exception('Not supported learning setting.')
    
    def test(self, data: Optional[Data] = None, prefix: str = '') -> Metrics:
        if (data is None or data == self.data) and self.learning_setting == 'transductive':
            data = self.data
        else:
            data = data.to(self.device, non_blocking=True)
            # data.x = self._encoder.predict(data)
            data = self.compute_aggregations(data)

        return super().test(data, prefix=prefix)

    def predict(self, data: Optional[Data] = None) -> torch.Tensor:
        if (data is None or data == self.data) and self.learning_setting == 'transductive':
            data = self.data
        else:
            # data.x = self._encoder.predict(data)
            data = self.compute_aggregations(data)

        return super().predict(data)

    '''
    def _gcn_layer(self, x: torch.Tensor, adj_t: SparseTensor) -> torch.Tensor:
        support = x @ self.weight
        output = matmul(adj_t, support)
        return output
    '''

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=2, dim=-1)

    def _projection(self, x: torch.Tensor) -> torch.Tensor:
        norms = torch.norm(x, p=2, dim=-1, keepdim=True)
        scaling_factor = torch.where(norms > 1, 1.0 / norms, 1.0)
        return x * scaling_factor

    def _aggregate(self, x: torch.Tensor, x_min_degree: torch.Tensor, layer_index: torch.Tensor) -> torch.Tensor:
        return x

    def pretrain_encoder(self, data: Data, prefix: str) -> Data:
        console.info('pretraining encoder')
        self._encoder.to(self.device)

        self.trainer.fit(
            model=self._encoder,
            epochs=self.encoder_epochs,
            optimizer=self.configure_encoder_optimizer(), 
            train_dataloader=self.data_loader(data, 'train'), 
            val_dataloader=self.data_loader(data, 'val'),
            test_dataloader=None,
            checkpoint=True,
            prefix=f'{prefix}encoder/',
        )

        self.trainer.reset()
        data.x = self._encoder.predict(data) 
        
        
        return data

    def compute_aggregations(self, data: Data) -> Data:

        with console.status('computing aggregations'):
            feature_initial = self.data.x
            adj_norm, self.x_min_degree = get_normalized_adj(data, flag_norm= 'True', fill_degree=self.fill_degree)
            adj_norm = adj_norm.to(self.device, non_blocking=True)
            x_min_degree = self.x_min_degree.to(self.device, non_blocking=True)
            x_list = [data.x]
            
            # print("debugging...data.x", data.x, "debugging...")
            x = F.normalize(data.x, p=2, dim=-1)
            # print("debugging...x", x, "debugging...")

            prev = torch.zeros_like(x)
            for layer_index, gcn_layer in enumerate(self.gcn_layers):
                # x = gcn_layer(x, adj_norm) + x_list[0] + prev # GCN layer
                x = self.bound_lipschitz * (self.alpha_1 * gcn_layer(x, adj_norm) + self.alpha_2 * torch.mean(x, dim=0)) + self.beta * feature_initial
                x = self._aggregate(x, x_min_degree, layer_index)  # aggregation
                prev = x
                
                if layer_index != len(self.gcn_layers) - 1:
                    x = self._projection(x)
                    # x = self._normalize(x)
                    #x_list.append(x)  
                else:    
                    x_list.append(x)
                x = self.bound_lipschitz * (self.alpha_1 * gcn_layer(x, adj_norm) + self.alpha_2 * torch.mean(x, dim=1, keepdim=True)) + self.beta * feature_initial

                # print("debugging...x", x, "debugging...") 
                # x = F.relu(x)

            # data.x = torch.stack(x_list, dim=-1) # TEST PARAMETER (w/o MLP)
            data.x = torch.stack([x], dim=-1)
        return data
    

    def configure_encoder_optimizer(self) -> Optimizer:
        Optim = {'sgd': SGD, 'adam': Adam}[self.optimizer_name]
        return Optim(self._encoder.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)