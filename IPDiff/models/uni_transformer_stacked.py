import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import radius_graph, knn_graph
from torch_scatter import scatter_softmax, scatter_sum
from typing import Union, Optional

from models.common import GaussianSmearing, MLP, batch_hybrid_edge_connection, outer_product
from models.uni_transformer import BaseX2HAttLayer, BaseH2XAttLayer


class PeriodicTransformHandler():
    """
    A class to handle periodic transformations.
    Args:
        periodic_dir: The periodicity vector.
        periodic_rot: The rotation matrix.
    
    Methods:
        periodic_shift: Shift the input tensor by the periodic vector.
        to: Move the periodic transform handler to the specified device.
    """
    def __init__(self, periodic_dir, periodic_rot):
        self.periodic_dir = torch.tensor(periodic_dir).to(torch.float32)
        self.periodic_rot = torch.tensor(periodic_rot).to(torch.float32)
        assert self.periodic_dir.shape == (3, ), "The periodicity vector must be a 3D vector"
        assert self.periodic_rot.shape == (3, 3), "The rotation matrix must be a 3x3 matrix"
        
    def periodic_shift(self, x: torch.Tensor, shift: Union[int, torch.Tensor]) -> torch.Tensor:
        """
        Shift the input tensor by the periodic vector.

        Args:
            x: The input tensor to shift.
            shift: The shift amount. 
                If int: Shift all elements in the input tensor by the same specified amount of the periodic vector. 
                If torch.Tensor: Shift the input tensor by the periodic vector for each element in the tensor.

        Returns:
            torch.Tensor: The shifted tensor.
        """
        if isinstance(shift, int):
            if shift == 0:
                return x
            return x @ torch.matrix_power(self.periodic_rot, shift).T + shift * self.periodic_dir
        else:
            assert x.shape[0] == shift.shape[0], "x and shift must have the same number of nodes"
            # sparsify shift to avoid repeated computation of matrix power
            shift_min, shift_max = shift.min(), shift.max()
            rotation_matrices_sparse = torch.zeros(3, 3, shift_max - shift_min + 1, device=x.device)
            for shift_idx in range(shift_min, shift_max + 1):
                rotation_matrices_sparse[:, :, shift_idx - shift_min] = torch.matrix_power(self.periodic_rot, shift_idx)
            rotation_matrices = rotation_matrices_sparse[:, :, shift - shift_min] # [3, 3, num_shifts]
            
            return torch.einsum('nx,Xxn->nX', x, rotation_matrices) + shift.unsqueeze(1) * self.periodic_dir
    
    def to(self, device):
        self.periodic_dir = self.periodic_dir.to(device)
        self.periodic_rot = self.periodic_rot.to(device)
        return self


class AttentionLayerO2TwoUpdateNodeGeneralStacked(nn.Module):
    def __init__(self, hidden_dim, n_heads, num_r_gaussian, edge_feat_dim, 
                 periodic_transform: Optional[PeriodicTransformHandler] = None,
                 act_fn='relu', norm=True,
                 num_x2h=1, num_h2x=1, r_min=0., r_max=10., num_node_types=8,
                 ew_net_type='r', x2h_out_fc=True, sync_twoup=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.edge_feat_dim = edge_feat_dim
        self.num_r_gaussian = num_r_gaussian
        self.norm = norm
        self.act_fn = act_fn
        self.num_x2h = num_x2h
        self.num_h2x = num_h2x
        self.r_min, self.r_max = r_min, r_max
        self.num_node_types = num_node_types
        self.ew_net_type = ew_net_type
        self.x2h_out_fc = x2h_out_fc
        self.sync_twoup = sync_twoup
        self.periodic_transform = periodic_transform

        self.distance_expansion = GaussianSmearing(self.r_min, self.r_max, num_gaussians=num_r_gaussian)

        self.x2h_layers = nn.ModuleList()
        for i in range(self.num_x2h):
            self.x2h_layers.append(
                BaseX2HAttLayer(hidden_dim, hidden_dim, hidden_dim, n_heads, edge_feat_dim,
                                r_feat_dim=num_r_gaussian * 4,
                                act_fn=act_fn, norm=norm,
                                ew_net_type=self.ew_net_type, out_fc=self.x2h_out_fc)
            )
        self.h2x_layers = nn.ModuleList()
        for i in range(self.num_h2x):
            self.h2x_layers.append(
                BaseH2XAttLayer(hidden_dim, hidden_dim, hidden_dim, n_heads, edge_feat_dim,
                                r_feat_dim=num_r_gaussian * 4,
                                act_fn=act_fn, norm=norm,
                                ew_net_type=self.ew_net_type)
            )

    def forward(self, h, x, edge_attr, edge_index, mask_ligand, edge_shift_label, e_w=None, fix_x=False):
        src, dst = edge_index
        if self.edge_feat_dim > 0:
            edge_feat = edge_attr  # shape: [#edges_in_batch, #bond_types]
        else:
            edge_feat = None
        
        # NOTE: use periodic transform handler to shift the input tensor
        if self.periodic_transform is not None:
            rel_x = x[dst] - self.periodic_transform.periodic_shift(x[src], edge_shift_label)
        else:
            rel_x = x[dst] - x[src]
        dist = torch.norm(rel_x, p=2, dim=-1, keepdim=True)

        h_in = h
        # 4 separate distance embedding for p-p, p-l, l-p, l-l
        for i in range(self.num_x2h):
            dist_feat = self.distance_expansion(dist)
            dist_feat = outer_product(edge_attr, dist_feat)
            h_out = self.x2h_layers[i](h_in, dist_feat, edge_feat, edge_index, e_w=e_w)
            h_in = h_out
        x2h_out = h_in

        new_h = h if self.sync_twoup else x2h_out
        for i in range(self.num_h2x):
            dist_feat = self.distance_expansion(dist)
            dist_feat = outer_product(edge_attr, dist_feat)
            delta_x = self.h2x_layers[i](new_h, rel_x, dist_feat, edge_feat, edge_index, e_w=e_w)
            if not fix_x:
                x = x + delta_x * mask_ligand[:, None]  # only ligand positions will be updated
            # NOTE: use periodic transform handler to shift the input tensor
            if self.periodic_transform is not None:
                rel_x = x[dst] - self.periodic_transform.periodic_shift(x[src], edge_shift_label)
            else:
                rel_x = x[dst] - x[src]
            dist = torch.norm(rel_x, p=2, dim=-1, keepdim=True)

        return x2h_out, x

class UniTransformerO2TwoUpdateGeneralStacked(nn.Module):
    def __init__(self, num_blocks, num_layers, hidden_dim, 
                 periodic_transform: Optional[PeriodicTransformHandler] = None,
                 n_heads=1, k=32,
                 num_r_gaussian=50, edge_feat_dim=0, num_node_types=8, act_fn='relu', norm=True,
                 cutoff_mode='radius', ew_net_type='r',
                 num_init_x2h=1, num_init_h2x=0, num_x2h=1, num_h2x=1, r_max=10., x2h_out_fc=True, sync_twoup=False):
        super().__init__()
        # Build the network
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.num_r_gaussian = num_r_gaussian
        self.edge_feat_dim = edge_feat_dim
        self.act_fn = act_fn
        self.norm = norm
        self.num_node_types = num_node_types
        # radius graph / knn graph
        self.cutoff_mode = cutoff_mode  # [radius, none]
        self.k = k
        self.ew_net_type = ew_net_type  # [r, m, none]
        
        # NOTE: register periodic transform handler to enable stacked generation
        self.periodic_transform = periodic_transform
        self.stack_gen = self.periodic_transform is not None

        self.num_x2h = num_x2h
        self.num_h2x = num_h2x
        self.num_init_x2h = num_init_x2h
        self.num_init_h2x = num_init_h2x
        self.r_max = r_max
        self.x2h_out_fc = x2h_out_fc
        self.sync_twoup = sync_twoup
        self.distance_expansion = GaussianSmearing(0., r_max, num_gaussians=num_r_gaussian)
        if self.ew_net_type == 'global':
            self.edge_pred_layer = MLP(num_r_gaussian, 1, hidden_dim)

        self.init_h_emb_layer = self._build_init_h_layer()
        self.base_block = self._build_share_blocks()
        
    
    def __repr__(self):
        return f'UniTransformerO2Stacked(num_blocks={self.num_blocks}, num_layers={self.num_layers}, n_heads={self.n_heads}, ' \
               f'act_fn={self.act_fn}, norm={self.norm}, cutoff_mode={self.cutoff_mode}, ew_net_type={self.ew_net_type}, ' \
               f'init h emb: {self.init_h_emb_layer.__repr__()} \n' \
               f'base block: {self.base_block.__repr__()} \n' \
               f'edge pred layer: {self.edge_pred_layer.__repr__() if hasattr(self, "edge_pred_layer") else "None"}) '

    def _build_init_h_layer(self):
        layer = AttentionLayerO2TwoUpdateNodeGeneralStacked(
            self.hidden_dim, self.n_heads, self.num_r_gaussian, self.edge_feat_dim, 
            periodic_transform=self.periodic_transform, act_fn=self.act_fn, norm=self.norm,
            num_x2h=self.num_init_x2h, num_h2x=self.num_init_h2x, r_max=self.r_max, num_node_types=self.num_node_types,
            ew_net_type=self.ew_net_type, x2h_out_fc=self.x2h_out_fc, sync_twoup=self.sync_twoup,
        )
        return layer

    def _build_share_blocks(self):
        # Equivariant layers
        base_block = []
        for l_idx in range(self.num_layers):
            layer = AttentionLayerO2TwoUpdateNodeGeneralStacked(
                self.hidden_dim, self.n_heads, self.num_r_gaussian, self.edge_feat_dim, 
                periodic_transform=self.periodic_transform, act_fn=self.act_fn, norm=self.norm,
                num_x2h=self.num_x2h, num_h2x=self.num_h2x, r_max=self.r_max, num_node_types=self.num_node_types,
                ew_net_type=self.ew_net_type, x2h_out_fc=self.x2h_out_fc, sync_twoup=self.sync_twoup,
            )
            base_block.append(layer)
        return nn.ModuleList(base_block)

    def _connect_edge(self, x, mask_ligand, batch, force_disable_stack_gen=False):
        # NOTE: this method is different from the original _connect_edge method in uni_transformer.py
        #       to enable edge construction for periodic structures
        # NOTE: force_disable_stack_gen is used to disable stacked generation for regular edge construction
        if force_disable_stack_gen or not self.stack_gen:
            if self.cutoff_mode == 'radius':
                edge_index = radius_graph(x, r=self.r, batch=batch, flow='source_to_target')
            elif self.cutoff_mode == 'knn':
                edge_index = knn_graph(x, k=self.k, batch=batch, flow='source_to_target')
            elif self.cutoff_mode == 'hybrid':
                edge_index = batch_hybrid_edge_connection(
                    x, k=self.k, mask_ligand=mask_ligand, batch=batch, add_p_index=True)
            else:
                raise ValueError(f'Not supported cutoff mode: {self.cutoff_mode}')
            return edge_index
        else:
            self.periodic_transform.to(x.device)
            num_nodes = x.shape[0]
            
            ### 1. Calculate the maximum shift required to cover all possible edges
            min_dist = torch.norm(self.periodic_transform.periodic_dir).item()
            x_proj = self.periodic_transform.periodic_dir @ x.T / min_dist
            # TODO: use scatter_max and scatter_min to enable different shifts for different batches
            max_dist = (torch.max(x_proj) - torch.min(x_proj)).item() + self.r_max
            max_shift = int(max_dist // min_dist)
            
            ### 2. Create a grid of points that covers all possible shifts
            x_all = x
            level_all = torch.zeros(num_nodes, dtype=torch.long, device=x.device)
            batch_all = batch
            mask_ligand_all = mask_ligand
            for shift in range(1, max_shift + 1):
                x_all = torch.cat([x_all, self.periodic_transform.periodic_shift(x, shift)], dim=0)
                level_all = torch.cat([level_all, torch.ones(num_nodes, dtype=torch.long, device=x.device) * shift], dim=0)
                batch_all = torch.cat([batch_all, batch], dim=0)
                mask_ligand_all = torch.cat([mask_ligand_all, mask_ligand], dim=0)
            
            ### 3. Sort all data by batch
            batch_all, batch_idx = torch.sort(batch_all)
            x_all = x_all[batch_idx]
            level_all = level_all[batch_idx]
            mask_ligand_all = mask_ligand_all[batch_idx]
            
            ### 4. Find edges via non-stacked connect edge function
            edge_index = self._connect_edge(x_all, mask_ligand_all, batch_all, force_disable_stack_gen=True)
            # check if all edges are within the same batch
            if not torch.all(batch_all[edge_index[0]] == batch_all[edge_index[1]]):
                raise ValueError('All edges must be within the same batch')
            
            ### 5. Clean up the edge index
            edge_index_clean = edge_index[:, torch.where(
                torch.logical_or(edge_index[0] < num_nodes, edge_index[1] < num_nodes)
            )[0]] # only keep edges with at least one node in the original points
            edge_shift_label = level_all[edge_index_clean[1]] - level_all[edge_index_clean[0]]
            edge_index_original = edge_index_clean % num_nodes
            
            # NOTE: debug code; currently we see edge distances become much smaller than regular atomic bond distances
            # import pdb; pdb.set_trace()
            # dist = torch.norm(x_all[edge_index_clean[0]] - x_all[edge_index_clean[1]], p=2, dim=-1, keepdim=True)
            # print('min edge distances: ')
            # print(f'within ligand: {torch.min(dist[torch.logical_and(mask_ligand_all[edge_index_clean[0]], mask_ligand_all[edge_index_clean[1]])])}')
            # print(f'between ligand and protein: {torch.min(dist[torch.logical_and(~mask_ligand_all[edge_index_clean[0]], mask_ligand_all[edge_index_clean[1]])])}')
            # print(f'between protein and ligand: {torch.min(dist[torch.logical_and(mask_ligand_all[edge_index_clean[0]], ~mask_ligand_all[edge_index_clean[1]])])}')
            # print(f'between protein and protein: {torch.min(dist[torch.logical_and(~mask_ligand_all[edge_index_clean[0]], ~mask_ligand_all[edge_index_clean[1]])])}')
            
            return edge_index_original, edge_shift_label
            

    @staticmethod
    def _build_edge_type(edge_index, mask_ligand):
        src, dst = edge_index
        edge_type = torch.zeros(len(src)).to(edge_index)
        n_src = mask_ligand[src] == 1
        n_dst = mask_ligand[dst] == 1
        edge_type[n_src & n_dst] = 0
        edge_type[n_src & ~n_dst] = 1
        edge_type[~n_src & n_dst] = 2
        edge_type[~n_src & ~n_dst] = 3
        edge_type = F.one_hot(edge_type, num_classes=4)
        return edge_type

    def forward(self, h, x, mask_ligand, batch, return_all=False, fix_x=False):

        all_x = [x]
        all_h = [h]

        for b_idx in range(self.num_blocks):
            if self.stack_gen:
                edge_index, edge_shift_label = self._connect_edge(x, mask_ligand, batch)
            else:
                edge_index = self._connect_edge(x, mask_ligand, batch)
            src, dst = edge_index

            # edge type (dim: 4)
            edge_type = self._build_edge_type(edge_index, mask_ligand)
            if self.ew_net_type == 'global':
                # NOTE: re-calculate distance in stacked radius graph
                dist = torch.norm(x[dst] - self.periodic_transform.periodic_shift(x[src], edge_shift_label), p=2, dim=-1, keepdim=True)
                dist_feat = self.distance_expansion(dist)
                logits = self.edge_pred_layer(dist_feat)
                e_w = torch.sigmoid(logits)
            else:
                e_w = None

            for l_idx, layer in enumerate(self.base_block):
                h, x = layer(h, x, edge_type, edge_index, mask_ligand, edge_shift_label, e_w=e_w, fix_x=fix_x)
            all_x.append(x)
            all_h.append(h)

        outputs = {'x': x, 'h': h}
        if return_all:
            outputs.update({'all_x': all_x, 'all_h': all_h})
        return outputs
