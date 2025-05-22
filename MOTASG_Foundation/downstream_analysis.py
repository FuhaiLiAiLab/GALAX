import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import (
    Linear,
    GCNConv,
    SAGEConv,
    GINConv,
    global_add_pool,
    global_mean_pool,
    global_max_pool
)

from torch_geometric.utils import add_self_loops, negative_sampling, degree
from torch_sparse import SparseTensor
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn.inits import glorot, zeros

import math
from tqdm.auto import tqdm
from typing import Optional, Tuple, Union
from torch import Tensor
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor, PairTensor, SparseTensor
from torch_geometric.utils import softmax
from torch_geometric.nn import aggr

import typing
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import (
    Adj,
    Size,
    NoneType,
    OptTensor,
    OptPairTensor,
    PairTensor,
    SparseTensor,
    torch_sparse,
)
from torch_geometric.utils import (
    add_self_loops,
    is_torch_sparse_tensor,
    remove_self_loops,
    softmax,
)
from torch_geometric.utils.sparse import set_sparse_value

if typing.TYPE_CHECKING:
    from typing import overload
else:
    from torch.jit import _overload_method as overload


def extract_attention_weights(attention_weights_sparse, reduction='mean'):
    """
    Extract edge weights from a SparseTensor representation of attention weights.
    
    Args:
        attention_weights_sparse (SparseTensor): The sparse tensor containing attention weights
        reduction (str): How to handle multiple attention heads: 'mean', 'sum', 'max', 'none',
                         or an integer to select a specific head
    
    Returns:
        tuple: (source_nodes, target_nodes, weights) where weights is a tensor of shape [num_edges]
               or [num_edges, num_heads] if reduction='none'
    """
    # Get row and column indices (source and target nodes)
    row, col = attention_weights_sparse.coo()[:2]
    
    # Get attention weights values
    attention_values = attention_weights_sparse.storage.value()
    
    # Apply reduction if needed
    if reduction == 'mean':
        attention_values = attention_values.mean(dim=1)
    elif reduction == 'sum':
        attention_values = attention_values.sum(dim=1)
    elif reduction == 'max':
        attention_values = attention_values.max(dim=1)[0]
    elif isinstance(reduction, int):
        # Select a specific attention head
        attention_values = attention_values[:, reduction]
    # If reduction is 'none', return all head weights
    
    return row, col, attention_values


class GATConv(MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = 'mean',
        bias: bool = True,
        residual: bool = False,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value
        self.residual = residual

        # In case we are operating in bipartite graphs, we apply separate
        # transformations 'lin_src' and 'lin_dst' to source and target nodes:
        self.lin = self.lin_src = self.lin_dst = None
        if isinstance(in_channels, int):
            self.lin = Linear(in_channels, heads * out_channels, bias=False,
                              weight_initializer='glorot')
        else:
            self.lin_src = Linear(in_channels[0], heads * out_channels, False,
                                  weight_initializer='glorot')
            self.lin_dst = Linear(in_channels[1], heads * out_channels, False,
                                  weight_initializer='glorot')

        # The learnable parameters to compute attention coefficients:
        self.att_src = Parameter(torch.empty(1, heads, out_channels))
        self.att_dst = Parameter(torch.empty(1, heads, out_channels))

        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False,
                                   weight_initializer='glorot')
            self.att_edge = Parameter(torch.empty(1, heads, out_channels))
        else:
            self.lin_edge = None
            self.register_parameter('att_edge', None)

        # The number of output channels:
        total_out_channels = out_channels * (heads if concat else 1)

        if residual:
            self.res = Linear(
                in_channels
                if isinstance(in_channels, int) else in_channels[1],
                total_out_channels,
                bias=False,
                weight_initializer='glorot',
            )
        else:
            self.register_parameter('res', None)

        if bias:
            self.bias = Parameter(torch.empty(total_out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        if self.lin is not None:
            self.lin.reset_parameters()
        if self.lin_src is not None:
            self.lin_src.reset_parameters()
        if self.lin_dst is not None:
            self.lin_dst.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        if self.res is not None:
            self.res.reset_parameters()
        glorot(self.att_src)
        glorot(self.att_dst)
        glorot(self.att_edge)
        zeros(self.bias)

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        size: Size = None,
        return_attention_weights: Optional[bool] = None,
    ) -> Union[
            Tensor,
            Tuple[Tensor, Tuple[Tensor, Tensor]],
            Tuple[Tensor, SparseTensor],
    ]:
        r"""Runs the forward pass of the module.

        Args:
            x (torch.Tensor or (torch.Tensor, torch.Tensor)): The input node
                features.
            edge_index (torch.Tensor or SparseTensor): The edge indices.
            edge_attr (torch.Tensor, optional): The edge features.
                (default: :obj:`None`)
            size ((int, int), optional): The shape of the adjacency matrix.
                (default: :obj:`None`)
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        # NOTE: attention weights will be returned whenever
        # `return_attention_weights` is set to a value, regardless of its
        # actual value (might be `True` or `False`). This is a current somewhat
        # hacky workaround to allow for TorchScript support via the
        # `torch.jit._overload` decorator, as we can only change the output
        # arguments conditioned on type (`None` or `bool`), not based on its
        # actual value.

        H, C = self.heads, self.out_channels

        res: Optional[Tensor] = None

        # We first transform the input node features. If a tuple is passed, we
        # transform source and target node features via separate weights:
        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in 'GATConv'"

            if self.res is not None:
                res = self.res(x)

            if self.lin is not None:
                x_src = x_dst = self.lin(x).view(-1, H, C)
            else:
                # If the module is initialized as bipartite, transform source
                # and destination node features separately:
                assert self.lin_src is not None and self.lin_dst is not None
                x_src = self.lin_src(x).view(-1, H, C)
                x_dst = self.lin_dst(x).view(-1, H, C)

        else:  # Tuple of source and target node features:
            x_src, x_dst = x
            assert x_src.dim() == 2, "Static graphs not supported in 'GATConv'"

            if x_dst is not None and self.res is not None:
                res = self.res(x_dst)

            if self.lin is not None:
                # If the module is initialized as non-bipartite, we expect that
                # source and destination node features have the same shape and
                # that they their transformations are shared:
                x_src = self.lin(x_src).view(-1, H, C)
                if x_dst is not None:
                    x_dst = self.lin(x_dst).view(-1, H, C)
            else:
                assert self.lin_src is not None and self.lin_dst is not None

                x_src = self.lin_src(x_src).view(-1, H, C)
                if x_dst is not None:
                    x_dst = self.lin_dst(x_dst).view(-1, H, C)

        x = (x_src, x_dst)

        # Next, we compute node-level attention coefficients, both for source
        # and target nodes (if present):
        alpha_src = (x_src * self.att_src).sum(dim=-1)
        alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
        alpha = (alpha_src, alpha_dst)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                # We only want to add self-loops for nodes that appear both as
                # source and target nodes:
                num_nodes = x_src.size(0)
                if x_dst is not None:
                    num_nodes = min(num_nodes, x_dst.size(0))
                num_nodes = min(size) if size is not None else num_nodes
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=self.fill_value,
                    num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = torch_sparse.set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form")

        # edge_updater_type: (alpha: OptPairTensor, edge_attr: OptTensor)
        alpha = self.edge_updater(edge_index, alpha=alpha, edge_attr=edge_attr,
                                  size=size)

        # propagate_type: (x: OptPairTensor, alpha: Tensor)
        out = self.propagate(edge_index, x=x, alpha=alpha, size=size)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if res is not None:
            out = out + res

        if self.bias is not None:
            out = out + self.bias

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                if is_torch_sparse_tensor(edge_index):
                    # TODO TorchScript requires to return a tuple
                    adj = set_sparse_value(edge_index, alpha)
                    return out, (adj, alpha)
                else:
                    return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out


    def edge_update(self, alpha_j: Tensor, alpha_i: OptTensor,
                    edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                    dim_size: Optional[int]) -> Tensor:
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        if index.numel() == 0:
            return alpha
        if edge_attr is not None and self.lin_edge is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
            alpha = alpha + alpha_edge
        
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, dim_size)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha

    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        return alpha.unsqueeze(-1) * x_j

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')

class GATv2Conv(MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = 'mean',
        bias: bool = True,
        share_weights: bool = False,
        residual: bool = False,
        **kwargs,
    ):
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value
        self.residual = residual
        self.share_weights = share_weights

        if isinstance(in_channels, int):
            self.lin_l = Linear(in_channels, heads * out_channels, bias=bias,
                                weight_initializer='glorot')
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = Linear(in_channels, heads * out_channels,
                                    bias=bias, weight_initializer='glorot')
        else:
            self.lin_l = Linear(in_channels[0], heads * out_channels,
                                bias=bias, weight_initializer='glorot')
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = Linear(in_channels[1], heads * out_channels,
                                    bias=bias, weight_initializer='glorot')

        self.att = Parameter(torch.empty(1, heads, out_channels))

        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False,
                                   weight_initializer='glorot')
        else:
            self.lin_edge = None

        # The number of output channels:
        total_out_channels = out_channels * (heads if concat else 1)

        if residual:
            self.res = Linear(
                in_channels
                if isinstance(in_channels, int) else in_channels[1],
                total_out_channels,
                bias=False,
                weight_initializer='glorot',
            )
        else:
            self.register_parameter('res', None)

        if bias:
            self.bias = Parameter(torch.empty(total_out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        if self.res is not None:
            self.res.reset_parameters()
        glorot(self.att)
        zeros(self.bias)

    def forward(  # noqa: F811
        self,
        x: Union[Tensor, PairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        return_attention_weights: Optional[bool] = None,
    ) -> Union[
            Tensor,
            Tuple[Tensor, Tuple[Tensor, Tensor]],
            Tuple[Tensor, SparseTensor],
    ]:
        r"""Runs the forward pass of the module.

        Args:
            x (torch.Tensor or (torch.Tensor, torch.Tensor)): The input node
                features.
            edge_index (torch.Tensor or SparseTensor): The edge indices.
            edge_attr (torch.Tensor, optional): The edge features.
                (default: :obj:`None`)
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        H, C = self.heads, self.out_channels

        res: Optional[Tensor] = None

        x_l: OptTensor = None
        x_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2

            if self.res is not None:
                res = self.res(x)

            x_l = self.lin_l(x).view(-1, H, C)
            if self.share_weights:
                x_r = x_l
            else:
                x_r = self.lin_r(x).view(-1, H, C)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2

            if x_r is not None and self.res is not None:
                res = self.res(x_r)

            x_l = self.lin_l(x_l).view(-1, H, C)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)

        assert x_l is not None
        assert x_r is not None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=self.fill_value,
                    num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = torch_sparse.set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form")

        # edge_updater_type: (x: PairTensor, edge_attr: OptTensor)
        alpha = self.edge_updater(edge_index, x=(x_l, x_r),
                                  edge_attr=edge_attr)

        # propagate_type: (x: PairTensor, alpha: Tensor)
        out = self.propagate(edge_index, x=(x_l, x_r), alpha=alpha)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if res is not None:
            out = out + res

        if self.bias is not None:
            out = out + self.bias

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                if is_torch_sparse_tensor(edge_index):
                    # TODO TorchScript requires to return a tuple
                    adj = set_sparse_value(edge_index, alpha)
                    return out, (adj, alpha)
                else:
                    return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def edge_update(self, x_j: Tensor, x_i: Tensor, edge_attr: OptTensor,
                    index: Tensor, ptr: OptTensor,
                    dim_size: Optional[int]) -> Tensor:
        x = x_i + x_j

        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            assert self.lin_edge is not None
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            x = x + edge_attr

        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, dim_size)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha

    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


class TransformerConv(MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        beta: bool = False,
        dropout: float = 0.,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        root_weight: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        self._alpha = None

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_key = Linear(in_channels[0], heads * out_channels, bias=bias)
        self.lin_query = Linear(in_channels[1], heads * out_channels,
                                bias=bias)
        self.lin_value = Linear(in_channels[0], heads * out_channels,
                                bias=bias)
        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.lin_edge = self.register_parameter('lin_edge', None)

        if concat:
            self.lin_skip = Linear(in_channels[1], heads * out_channels,
                                   bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * heads * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)
        else:
            self.lin_skip = Linear(in_channels[1], out_channels, bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if self.edge_dim:
            self.lin_edge.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()

    def forward(  # noqa: F811
        self,
        x: Union[Tensor, PairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        return_attention_weights: Optional[bool] = None,
    ) -> Union[
            Tensor,
            Tuple[Tensor, Tuple[Tensor, Tensor]],
            Tuple[Tensor, SparseTensor],
    ]:
        r"""Runs the forward pass of the module.

        Args:
            x (torch.Tensor or (torch.Tensor, torch.Tensor)): The input node
                features.
            edge_index (torch.Tensor or SparseTensor): The edge indices.
            edge_attr (torch.Tensor, optional): The edge features.
                (default: :obj:`None`)
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        H, C = self.heads, self.out_channels

        if isinstance(x, Tensor):
            x = (x, x)

        query = self.lin_query(x[1]).view(-1, H, C)
        key = self.lin_key(x[0]).view(-1, H, C)
        value = self.lin_value(x[0]).view(-1, H, C)

        # propagate_type: (query: Tensor, key:Tensor, value: Tensor,
        #                  edge_attr: OptTensor)
        out = self.propagate(edge_index, query=query, key=key, value=value,
                             edge_attr=edge_attr)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.root_weight:
            x_r = self.lin_skip(x[1])
            if self.lin_beta is not None:
                beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1))
                beta = beta.sigmoid()
                out = beta * x_r + (1 - beta) * out
            else:
                out = out + x_r

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out


    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        if self.lin_edge is not None:
            assert edge_attr is not None
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads,
                                                      self.out_channels)
            key_j = key_j + edge_attr

        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = value_j
        if edge_attr is not None:
            out = out + edge_attr

        out = out * alpha.view(-1, self.heads, 1)
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


def to_sparse_tensor(edge_index, num_nodes):
    return SparseTensor.from_edge_index(
        edge_index, sparse_sizes=(num_nodes, num_nodes)
    ).to(edge_index.device)


def creat_gnn_layer(name, first_channels, second_channels, heads):
    if name == "sage":
        layer = SAGEConv(first_channels, second_channels)
    elif name == "gcn":
        layer = GCNConv(first_channels, second_channels)
    elif name == "gin":
        layer = GINConv(Linear(first_channels, second_channels), train_eps=True)
    elif name == "gat":
        layer = GATConv(-1, second_channels, heads=heads)
    elif name == "gat2":
        layer = GATv2Conv(-1, second_channels, heads=heads)
    elif name == "transformer":
        layer = TransformerConv(first_channels, second_channels, heads=heads, beta=True, dropout=0.1)
    else:
        raise ValueError(name)
    return layer


def create_input_layer(num_nodes, num_node_feats,
                       use_node_feats=True, node_emb=None):
    emb = None
    if use_node_feats:
        input_dim = num_node_feats
        if node_emb:
            emb = torch.nn.Embedding(num_nodes, node_emb)
            input_dim = input_dim + node_emb
    else:
        emb = torch.nn.Embedding(num_nodes, node_emb)
        input_dim = node_emb
    return input_dim, emb


def creat_activation_layer(activation):
    if activation is None:
        return nn.Identity()
    elif activation == "relu":
        return nn.ReLU(inplace=False)
    elif activation == "elu":
        return nn.ELU(inplace=False)
    elif activation == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.2, inplace=False)
    else:
        raise ValueError("Unknown activation")


class DSGNNEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers=2,
        dropout=0.5,
        bn=False,
        layer="gcn",
        activation="elu",
        use_node_feats=True,
        num_nodes=None,
        node_emb=None,
    ):

        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        bn = nn.BatchNorm1d if bn else nn.Identity
        self.use_node_feats = use_node_feats
        self.node_emb = node_emb

        if node_emb is not None and num_nodes is None:
            raise RuntimeError("Please provide the argument `num_nodes`.")

        in_channels, self.emb = create_input_layer(
            num_nodes, in_channels, use_node_feats=use_node_feats, node_emb=node_emb
        )
        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels
            heads = 1 if i == num_layers - 1 or 'gat' not in layer else 4

            self.convs.append(creat_gnn_layer(layer, first_channels, second_channels, heads))
            self.bns.append(bn(second_channels*heads))

        self.dropout = nn.Dropout(dropout)
        self.activation = creat_activation_layer(activation)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

        for bn in self.bns:
            if not isinstance(bn, nn.Identity):
                bn.reset_parameters()

        if self.emb is not None:
            nn.init.xavier_uniform_(self.emb.weight)

    def create_input_feat(self, x):
        if self.use_node_feats:
            input_feat = x
            if self.node_emb:
                input_feat = torch.cat([self.emb.weight, input_feat], dim=-1)
        else:
            input_feat = self.emb.weight
        return input_feat

    def forward(self, x, edge_index, sample_id=None, return_attention_weights=False, reduction='mean'):
        x = self.create_input_feat(x)
        edge_index = to_sparse_tensor(edge_index, x.size(0))
        
        attention_weights_dict = {}
        for i, conv in enumerate(self.convs[:-1]):
            x = self.dropout(x)
            x, attention_weights = conv(x, edge_index, return_attention_weights=True)
            # Extract source, target, and weights
            src_nodes, dst_nodes, edge_weights = extract_attention_weights(attention_weights, reduction)
            # Store in dictionary with more meaningful structure
            key = f"layer_{i}_sample_{sample_id}"
            attention_weights_dict[key] = {
                'source': src_nodes,
                'target': dst_nodes,
                'weights': edge_weights
            }
            x = self.bns[i](x)
            x = self.activation(x)
        
        x = self.dropout(x)
        x, last_attention_weights = self.convs[-1](x, edge_index, return_attention_weights=True)
        # Extract source, target, and weights for the last layer
        src_nodes, dst_nodes, last_attention_weights = extract_attention_weights(last_attention_weights)
        # Store in dictionary with more meaningful structure
        key = f"layer_{len(self.convs)-1}_sample_{sample_id}"
        attention_weights_dict[key] ={
            'source': src_nodes,
            'target': dst_nodes,
            'weights': last_attention_weights
        }
        x = self.bns[-1](x)
        x = self.activation(x)
        
        return x, attention_weights_dict


# Define the Attentive Readout module
class AttentiveReadout(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(AttentiveReadout, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Tanh(),
            nn.Linear(hidden_channels, 1)
        )

    def reset_parameters(self):
        # Reset the linear layers in the sequential module
        for module in self.gate:
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
    
    def forward(self, x):
        # x has shape: (batch_size, num_nodes, in_channels)
        attn_scores = self.gate(x)               # shape: (B, N, 1)
        attn_weights = F.softmax(attn_scores, dim=1)  # Normalize across nodes
        # Weighted sum: sum_{i=1}^{N} attn_weights[i]*x[i]
        out = (attn_weights * x).sum(dim=1)       # shape: (B, in_channels)
        return out

class MOTASG_KO_Reg(nn.Module):
    def __init__(
        self,
        text_input_dim,
        omic_input_dim,
        pre_input_dim,
        fusion_dim,
        internal_graph_output_dim,
        graph_output_dim,
        linear_input_dim,
        linear_hidden_dim,
        linear_output_dim,
        num_entity,
        text_encoder,
        encoder,
        internal_encoder,
        leaky_relu_slope=0.3,
        dropout_rate=0.1
    ):
        super().__init__()

        self.text_encoder = text_encoder
        self.encoder = encoder
        self.internal_encoder = internal_encoder

        self.name_linear_transform = nn.Linear(text_input_dim, text_input_dim)
        self.desc_linear_transform = nn.Linear(text_input_dim, text_input_dim)
        self.omic_linear_transform = nn.Linear(omic_input_dim, omic_input_dim)
        self.cross_modal_fusion = nn.Linear(text_input_dim * 2 + omic_input_dim, fusion_dim)
        self.pre_transform = nn.Linear(pre_input_dim, internal_graph_output_dim)

        self.act = nn.LeakyReLU(negative_slope=leaky_relu_slope)
        self.dropout = nn.Dropout(dropout_rate)

        ### Attentive Readout
        self.attentive_readout = AttentiveReadout(graph_output_dim, graph_output_dim)

        # # Simple aggregations
        # self.mean_aggr = aggr.MeanAggregation()
        # self.max_aggr = aggr.MaxAggregation()
        # # Learnable aggregations
        # self.softmax_aggr = aggr.SoftmaxAggregation(learn=True)
        # self.powermean_aggr = aggr.PowerMeanAggregation(learn=True)

        self.regression = nn.Linear(linear_input_dim, 1)

        # # Linear representation
        # self.lin_transform = nn.Linear(graph_output_dim, 1)
        # self.linear_repr = nn.Sequential(
        #     nn.Linear(num_entity, linear_input_dim),
        #     self.act,
        #     nn.Dropout(dropout_rate),
        #     nn.Linear(linear_input_dim, linear_hidden_dim),
        #     self.act,
        #     nn.Dropout(dropout_rate),
        #     nn.Linear(linear_hidden_dim, linear_output_dim)
        # )

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.internal_encoder.reset_parameters()

        # Reset linear layers.
        self.name_linear_transform.reset_parameters()
        self.desc_linear_transform.reset_parameters()
        self.omic_linear_transform.reset_parameters()
        self.cross_modal_fusion.reset_parameters()
        self.pre_transform.reset_parameters()

        if hasattr(self.attentive_readout, 'reset_parameters'):
            self.attentive_readout.reset_parameters()

        if isinstance(self.regression, nn.Sequential):
            for module in self.regression:
                if hasattr(module, 'reset_parameters'):
                    module.reset_parameters()
        else:   
            self.regression.reset_parameters()
        
        # self.lin_transform.reset_parameters()
        
        # # Reset Sequential modules properly by resetting each component
        # for module in self.linear_repr:
        #     if hasattr(module, 'reset_parameters'):
        #         module.reset_parameters()

    def forward(self, x, pre_x, edge_index, internal_edge_index, ppi_edge_index,
                num_entity, name_embeddings, desc_embeddings, batch_size, sample_id):
        
        num_node = num_entity * batch_size

        # ********************** Cross-modality Fusion ***************************
        name_emb = self.name_linear_transform(name_embeddings).clone()
        name_emb = self.act(name_emb.repeat(batch_size, 1))
        desc_emb = self.desc_linear_transform(desc_embeddings).clone()
        desc_emb = self.act(desc_emb.repeat(batch_size, 1))
        omic_emb = self.act(self.omic_linear_transform(x).clone())
        merged_emb = torch.cat([name_emb, desc_emb, omic_emb], dim=-1)
        cross_x = self.cross_modal_fusion(merged_emb)
        # ************************************************************************

        ko_feature = torch.zeros(x.shape[0], 1, device=cross_x.device)
        x = torch.cat([x, ko_feature], dim=-1)
        cross_x = torch.cat([cross_x, ko_feature], dim=-1)
        pre_x = torch.cat([pre_x, ko_feature], dim=-1)

        # *********************** Internal Graph Encoder *************************
        # Check if internal_encoder returns attention weights and handle accordingly
        internal_output = self.internal_encoder(cross_x, internal_edge_index)
        # If it returns a tuple (features, attention_weights)
        if isinstance(internal_output, tuple):
            internal_z = internal_output[0]  # Extract just the node features
        else:
            internal_z = internal_output
        z = self.act(internal_z) + x
        # ************************************************************************
        pre_x_transformed = self.pre_transform(pre_x)
        z = z + pre_x_transformed
        # *************************** Graph Encoder ******************************
        return_attention_weights = True
        reduction = 'mean'
        z, attention_weights_dict = self.encoder(z, edge_index, sample_id=sample_id, return_attention_weights=return_attention_weights, reduction=reduction)
        # ************************************************************************
        return z, attention_weights_dict