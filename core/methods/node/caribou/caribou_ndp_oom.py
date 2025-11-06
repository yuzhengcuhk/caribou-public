import numpy as np
import torch
import math
from typing import Annotated, Literal, Union
from torch_geometric.data import Data
from torch_sparse import SparseTensor, matmul
from core import console
from core.args.utils import ArgInfo
from core.methods.node import Caribou
from core.privacy.algorithms import BoundPMA
from torch import nn
import torch.nn.functional as F
from core.privacy.algorithms import PMA, NoisySGD
from core.methods.node.caribou.caribou_inf import get_normalized_adj
from core.data.transforms import BoundOutDegree
from core.privacy.mechanisms import ComposedNoisyMechanism
from core.modules.base import Metrics, Stage
from opacus.optimizers import DPOptimizer
from core.data.loader import NodeDataLoader

class NodePrivCaribou (Caribou):
    """node-private Caribou method"""

    def __init__(self,
                num_classes,
                epsilon:       Annotated[float, ArgInfo(help='DP epsilon parameter', option='-e')],
                delta:         Annotated[Union[Literal['auto'], float], 
                                                 ArgInfo(help='DP delta parameter (if "auto", sets a proper value based on data size)', option='-d')] = 'auto',
                max_degree:    Annotated[int,   ArgInfo(help='max degree to sample per each node')] = 100,
                max_grad_norm: Annotated[float, ArgInfo(help='maximum norm of the per-sample gradients')] = 1.0,
                batch_size:    Annotated[int,   ArgInfo(help='batch size')] = 256,
                **kwargs:      Annotated[dict,  ArgInfo(help='extra options passed to base class', bases=[Caribou], exclude=['batch_norm'])]
                ):

        super().__init__(num_classes, 
                        batch_norm=False, 
                        batch_size=batch_size, 
                        **kwargs)
        self.epsilon = epsilon
        self.delta = delta
        # self.num_edges = None  # will be used to set delta if it is 'auto'
        self.max_degree = max_degree
        self.max_grad_norm = max_grad_norm
        

        self.num_train_nodes = None  # will be used to set delta if it is 'auto'

    def calibrate(self):
        
        self.hops = min(self.hops,  (1 - self.bound_lipschitz ** self.hops) * (1 + self.bound_lipschitz)/(1 - self.bound_lipschitz)/ (1 + self.bound_lipschitz ** self.hops)) 
        print("debugging...hops", self.hops, "debugging...")
        self.pma_mechanism = BoundPMA(noise_scale=0.0, hops=self.hops)
        # self.pma_mechanism = PMA(noise_scale=0.0, hops=self.hops)

        self.encoder_noisy_sgd = NoisySGD(
            noise_scale=0.0, 
            dataset_size=self.num_train_nodes, 
            batch_size=self.batch_size, 
            epochs=self.encoder_epochs,
            max_grad_norm=self.max_grad_norm,
            replacement=True
        )

        self.classifier_noisy_sgd = NoisySGD(
            noise_scale=0.0, 
            dataset_size=self.num_train_nodes, 
            batch_size=self.batch_size, 
            epochs=self.epochs,
            max_grad_norm=self.max_grad_norm,
            replacement=True
        )

        composed_mechanism = ComposedNoisyMechanism(
            noise_scale=0.0,
            mechanism_list=[
                self.encoder_noisy_sgd, 
                self.pma_mechanism, 
                self.classifier_noisy_sgd
            ]
        )

        with console.status('calibrating noise to privacy budget'):
            if self.delta == 'auto':
                delta = 0.0 if np.isinf(self.epsilon) else 1. / (10 ** len(str(self.num_train_nodes)))
                console.info('delta = %.0e' % delta)
            
            self.noise_scale = composed_mechanism.calibrate(eps=self.epsilon, delta=delta)
            console.info(f'noise scale: {self.noise_scale:.4f}\n')

        self._encoder = self.encoder_noisy_sgd.prepare_module(self._encoder)
        self._classifier = self.classifier_noisy_sgd.prepare_module(self._classifier)

    def fit(self, data: Data, learn: str = 'transductive', prefix: str = '') -> Metrics:
        num_train_nodes = data.train_mask.sum().item()

        if num_train_nodes != self.num_train_nodes:
            self.num_train_nodes = num_train_nodes
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
        # sensitivity = 1
        # sensitivity = torch.tensor(sensitivity).to(self.device)

        print("self.x_min_degree:", self.x_min_degree)

        if 1 <= self.x_min_degree <= 3:
            C_min = 0.1584
        elif self.x_min_degree > 3:
            C_min = self.x_min_degree / math.sqrt(self.x_min_degree + 1) - self.x_min_degree / math.sqrt(self.x_min_degree + 2)  # This is the default/standard case in our paper
        else:
            raise ValueError('x_min_degree should be greater than 0')
        
        sensitivity = 1 + 2 * self.bound_lipschitz * self.alpha_2 * node_num / (node_num + 1) + self.alpha_1 * self.bound_lipschitz * (math.sqrt(self.max_degree) / (self.x_min_degree + 1) / (self.x_min_degree + 2) + C_min * math.sqrt(self.max_degree) / math.sqrt(self.max_degree + 1) + 1 / math.sqrt(self.x_min_degree + 2)) 

        
        x = self.pma_mechanism(x, sensitivity)

        return x 