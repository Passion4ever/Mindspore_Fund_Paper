import numpy as np
import mindspore as ms
from mindspore import nn
import mindspore.ops as ops
from mindspore_gl.nn import GINConv
from mindspore_gl.nn import global_mean_pool as gap, global_max_pool as gmp
from mindspore_gl.utils import k_hop_subgraph
import copy

def Subgraph(data, aug_ratio):
    data = copy.deepcopy(data)

    x = data.x
    edge_index = data.edge_index

    sub_num = int(data.num_nodes * aug_ratio)
    idx_sub = ms.Tensor(np.random.randint(0, data.num_nodes, (1,)), ms.int32).to_tensor()
    last_idx = idx_sub
    diff = None

    for k in range(1, sub_num):
        keep_idx, _, _, _ = k_hop_subgraph(last_idx, 1, edge_index)
        if keep_idx.shape[0] == last_idx.shape[0] or keep_idx.shape[0] >= sub_num or k == sub_num - 1:
            combined = ops.Concat()((last_idx, keep_idx))
            uniques, counts = ops.UniqueWithCounts()(combined)
            diff = uniques[counts == 1]
            break

        last_idx = keep_idx

    diff_keep_num = min(sub_num - last_idx.shape[0], diff.shape[0])
    diff_keep_idx = ops.RandomShuffle()(diff)[0][:diff_keep_num]
    final_keep_idx = ops.Concat()((last_idx, diff_keep_idx))

    drop_idx = ms.Tensor(np.ones(x.shape[0], dtype=bool), ms.bool_)
    drop_idx[final_keep_idx] = False
    x[drop_idx] = 0

    edge_index, _ = Subgraph(final_keep_idx, edge_index)

    data.x = x
    data.edge_index = edge_index
    return data

class CMMS_GCL(nn.Cell):
    def __init__(self, num_features_xd=84, dropout=0.2, aug_ratio=0.4):
        super(CMMS_GCL, self).__init__()

        self.W_rnn = nn.GRU(input_size=100, hidden_size=100, num_layers=1, bidirectional=True)

        self.fc = nn.SequentialCell([
            nn.Dense(200, 512),
            nn.ReLU(),
            nn.Dense(512, 256)
        ])

        self.linear = nn.SequentialCell([
            nn.Dense(200, 512),
            nn.Dense(512, 256)
        ])

        self.fc_g = nn.SequentialCell([
            nn.Dense(num_features_xd * 10 * 2, 1024),
            nn.ReLU(),
            nn.Dropout(keep_prob=1-dropout),
            nn.Dense(1024, 512)
        ])

        self.fc_final = nn.SequentialCell([
            nn.Dense(768, 256),
            nn.ReLU(),
            nn.Dropout(keep_prob=1-dropout),
            nn.Dense(256, 1)
        ])

        self.conv1 = GINConv(nn.Dense(num_features_xd, num_features_xd))
        self.conv2 = GINConv(nn.Dense(num_features_xd, num_features_xd * 10))
        self.relu = nn.ReLU()
        self.aug_ratio = aug_ratio

    def construct(self, data, x, edge_index, batch, smi_em):
        # Sequence Encoder
        smi_em = ops.Reshape()(smi_em, (-1, 100, 100))
        smi_em, _ = self.W_rnn(smi_em)
        smi_em = ops.ReLU()(smi_em)
        sentence_att = self.softmax(ops.Tanh()(self.fc(smi_em)), 1)
        smi_em = ops.ReduceSum(True)(ops.BatchMatMul()(sentence_att.transpose(0, 2, 1), smi_em), 1) / 10
        smi_em = self.linear(smi_em)

        # Graph Encoder
        x_g = self.relu(self.conv1(x, edge_index))
        x_g = self.relu(self.conv2(x_g, edge_index))
        x_g = ops.Concat(1)((gmp(x_g, batch), gap(x_g, batch)))
        x_g = self.fc_g(x_g)

        # Sub-structure Sampling
        data_aug1 = Subgraph(data, self.aug_ratio)
        y, y_edge_index, y_batch = data_aug1.x, data_aug1.edge_index, data_aug1.batch

        y_g = self.relu(self.conv1(y, edge_index))
        y_g = self.relu(self.conv2(y_g, edge_index))
        y_g = ops.Concat(1)((gmp(y_g, batch), gap(y_g, batch)))
        y_g = self.fc_g(y_g)

        # Stability predictor
        z = self.fc_final(ops.Concat(1)((x_g, smi_em)))
        return z, x_g, y_g

    @staticmethod
    def softmax(input, axis=1):
        input_size = input.shape
        trans_input = ops.Transpose()(input, (axis, len(input_size) - 1))
        soft_max_2d = ops.Softmax()(ops.Reshape()(trans_input, (-1, trans_input.shape[-1])))
        return ops.Transpose()(ops.Reshape()(soft_max_2d, trans_input.shape), (axis, len(input_size) - 1))
