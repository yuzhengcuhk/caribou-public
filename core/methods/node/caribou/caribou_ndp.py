import numpy as np
import torch
from typing import Annotated, Literal, Union
import math
from torch_geometric.data import Data
from torch_sparse import SparseTensor, matmul
from core import console
from core.args.utils import ArgInfo
from core.methods.node import Caribou
from core.privacy.algorithms import BoundPMA
from core.modules.base import Metrics
from torch import nn
import torch.nn.functional as F
from core.data.transforms import BoundOutDegree
from core.methods.node.caribou.caribou_inf import get_normalized_adj


class NodePrivCaribou (Caribou):
    """node-private Caribou method"""

    def __init__(self,
                 num_classes,
                 epsilon:       Annotated[float, ArgInfo(help='DP epsilon parameter', option='-e')],
                 delta:         Annotated[Union[Literal['auto'], float], 
                                                 ArgInfo(help='DP delta parameter (if "auto", sets a proper value based on data size)', option='-d')] = 'auto',
                 max_degree:    Annotated[int,   ArgInfo(help='max degree to sample per each node')] = 100,
                 **kwargs:      Annotated[dict,  ArgInfo(help='extra options passed to base class', bases=[Caribou])]
                 ):

        super().__init__(num_classes, **kwargs)
        self.epsilon = epsilon
        self.delta = delta
        self.num_edges = None  # will be used to set delta if it is 'auto'
        self.max_degree = max_degree

    def calibrate(self):
        
        self.hops = min(self.hops,  (1 - self.bound_lipschitz ** self.hops) * (1 + self.bound_lipschitz)/(1 - self.bound_lipschitz)/ (1 + self.bound_lipschitz ** self.hops)) 
        self.pma_mechanism = BoundPMA(noise_scale=0.0, hops=self.hops)
        
        with console.status('calibrating noise to privacy budget'):
            if self.delta == 'auto':
                delta = 0.0 if np.isinf(self.epsilon) else 1. / (10 ** len(str(self.num_edges)))
                console.info('delta = %.0e' % delta)

            self.noise_scale =  self.pma_mechanism.calibrate(eps=self.epsilon, delta=delta)
            console.info(f'noise scale: {self.noise_scale:.4f}\n')


    def fit(self, data: Data, learn: str = 'transductive', prefix: str = '') -> Metrics:
        if data.num_edges != self.num_edges:
            self.num_edges = data.num_edges
            self.calibrate()

        return super().fit(data, learn=learn, prefix=prefix)

    def compute_aggregations(self, data: Data) -> Data:
        with console.status('bounding the number of neighbors per node'):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            data = BoundOutDegree(self.max_degree)(data)
            # assert torch.all(data.adj_t.sum(0)<=self.max_degree)
        return super().compute_aggregations(data)

    def _aggregate(self, x: torch.Tensor, x_min_degree: torch.Tensor, layer_index: torch.Tensor) -> torch.Tensor:

        x = x.clone().detach().to(self.device)
        node_num = torch.tensor(x.shape[0]).to(self.device)
        # x_min_degree = x_min_degree.to(self.device)
        
        # sensitivity = 1
        # sensitivity = torch.tensor(sensitivity).to(self.device)

        if 1 <= self.x_min_degree <= 3:
            C_min = 0.1584
        elif self.x_min_degree > 3:
            C_min = self.x_min_degree / math.sqrt(self.x_min_degree + 1) - self.x_min_degree / math.sqrt(self.x_min_degree + 2)  # This is the default/standard case in our paper
        else:
            C_min = 0.1584
            # raise ValueError('x_min_degree should be greater than 0')
        sensitivity = 1 + 2 * self.bound_lipschitz * self.alpha_2 * node_num / (node_num + 1) + self.alpha_1 * self.bound_lipschitz * (math.sqrt(self.max_degree) / (self.x_min_degree + 1) / (self.x_min_degree + 2) + C_min * math.sqrt(self.max_degree) / math.sqrt(self.max_degree + 1) + 1 / math.sqrt(self.x_min_degree + 2)) 
        
        x = self.pma_mechanism(x, sensitivity)
    
    
        return x
