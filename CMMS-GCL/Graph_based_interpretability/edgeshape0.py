
import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
from numpy.random import default_rng



def edgeshaper(model, data, x, E, batch, smi_em, M=100, target_class=0, P=None, log_odds=True, seed=42):
    rng = default_rng(seed=seed)
    model.set_train(mode=False)
    phi_edges = []
    num_nodes = x.shape[0]
    num_edges = E.shape[1]


    if P is None:
        max_num_edges = num_nodes * (num_nodes - 1)
        graph_density = num_edges / max_num_edges
        P = graph_density

    for j in range(num_edges):
        marginal_contrib = 0
        for i in range(M):
            E_z_mask = rng.binomial(1, P, num_edges).astype(np.float32)
            E_mask = np.ones(num_edges, dtype=np.float32)

            
            pi = rng.permutation(num_edges).astype(np.int32)

            
            E_j_plus_index = np.ones(num_edges, dtype=np.int32)
            E_j_minus_index = np.ones(num_edges, dtype=np.int32)
            selected_edge_index = np.where(pi == j)[0].item()
            for k in range(num_edges):
                if k <= selected_edge_index:
                    E_j_plus_index[pi[k]] = E_mask[pi[k]]
                else:
                    E_j_plus_index[pi[k]] = E_z_mask[pi[k]]

            for k in range(num_edges):
                if k < selected_edge_index:
                    E_j_minus_index[pi[k]] = E_mask[pi[k]]
                else:
                    E_j_minus_index[pi[k]] = E_z_mask[pi[k]]

            
            retained_indices_plus = Tensor(pi[E_j_plus_index.astype(np.bool)], dtype=mindspore.int32)
            E_j_plus = ops.Gather()(E, retained_indices_plus, 1)

            out = model(data, x, E_j_plus, batch, smi_em)

            if not log_odds:
                out_prob = nn.Softmax()(out)
            else:
                out_prob = out  # out prob variable now contains log_odds

            V_j_plus = out_prob[:, target_class].asnumpy().item()

            retained_indices_minus = Tensor(pi[E_j_minus_index.astype(np.bool)], dtype=mindspore.int32)
            E_j_minus = ops.Gather()(E, retained_indices_minus, 1)

            out = model(data, x, E_j_minus, batch, smi_em)

            if not log_odds:
                out_prob = nn.Softmax()(out)
            else:
                out_prob = out

            V_j_minus = out_prob[:, target_class].asnumpy().item()

            marginal_contrib += (V_j_plus - V_j_minus)

        phi_edges.append(marginal_contrib / M)
    return phi_edges
