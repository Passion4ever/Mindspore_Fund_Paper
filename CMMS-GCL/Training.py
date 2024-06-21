import sys
from model import *
from utils import *
from evalution import *
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, Parameter, context
from mindspore.train import Model
from mindspore.train.callback import LossMonitor
from mindspore.dataset import GeneratorDataset

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

def ViewContrastiveLoss(view_i, view_j, batch, temperature):
    z_i = ops.L2Normalize(axis=1)(view_i)
    z_j = ops.L2Normalize(axis=1)(view_j)

    representations = ops.Concat(axis=0)((z_i, z_j))
    similarity_matrix = ops.MatMul(transpose_b=True)(representations, representations)
    sim_ij = ops.Eye()(batch, batch, ms.float32) * similarity_matrix[:batch, batch:]
    sim_ji = ops.Eye()(batch, batch, ms.float32) * similarity_matrix[batch:, :batch]
    positives = ops.Concat(axis=0)((sim_ij, sim_ji))

    nominator = ops.Exp()(positives / temperature)
    negatives_mask = ops.Ones()((2 * batch, 2 * batch), ms.float32) - ops.Eye()((2 * batch, 2 * batch), ms.float32)
    denominator = negatives_mask * ops.Exp()(similarity_matrix / temperature)

    loss_partial = -ops.Log()(nominator / ops.ReduceSum(True)(denominator, axis=1))
    loss = ops.ReduceSum(True)(loss_partial) / (2 * batch)
    return loss

def train_step(model, data, optimizer):
    model.set_train()
    n = data.y.shape[0]  # batch
    output, x_g, y_g = model(data, data.x, data.edge_index, data.batch, data.smi_em)
    loss_1 = criterion(output, data.y)
    T = 0.2
    loss_2 = ViewContrastiveLoss(x_g, y_g, n, T)
    loss = loss_1 + 0.1 * loss_2
    grads = ops.GradOperation(get_by_list=True)(model, optimizer.parameters)(data, loss)
    optimizer(grads)
    return loss

def train(model, train_loader, optimizer, epoch):
    print(f'Training on {len(train_loader)} samples...')
    for batch_idx, data in enumerate(train_loader.create_tuple_iterator()):
        loss = train_step(model, data, optimizer)

        if batch_idx % LOG_INTERVAL == 0:
            print(f'Train epoch: {epoch} [{batch_idx * len(data.x)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.asnumpy():.6f}')

def predicting(model, loader):
    model.set_train(False)
    total_preds = []
    total_labels = []
    for data in loader.create_tuple_iterator():
        output, _, _ = model(data, data.x, data.edge_index, data.batch, data.smi_em)
        preds = output.asnumpy()
        labels = data.y.asnumpy()
        total_preds.append(preds)
        total_labels.append(labels)
    return np.concatenate(total_labels).flatten(), np.concatenate(total_preds).flatten()

if __name__ == "__main__":
    cuda_name = "GPU:" + str(int(sys.argv[3]))
    context.set_context(device_target="GPU", device_id=int(sys.argv[3]))
    print('cuda_name:', cuda_name)

    TRAIN_BATCH_SIZE = 256
    TEST_BATCH_SIZE = 256
    LR = 0.0005
    LOG_INTERVAL = 20
    NUM_EPOCHS = 200

    print('Learning rate: ', LR)
    print('Epochs: ', NUM_EPOCHS)

    processed_train = 'data/processed/' + 'train.pt'
    processed_test = 'data/processed/' + 'test.pt'
    if ((not os.path.isfile(processed_train)) or (not os.path.isfile(processed_test))):
        print('please run create_data.py to prepare data in mindspore format!')
    else:
        train_data = TestbedDataset(root='data', dataset='train')
        test_data = TestbedDataset(root='data', dataset='test')

        train_loader = GeneratorDataset(train_data, ['data'], shuffle=True).batch(TRAIN_BATCH_SIZE)
        test_loader = GeneratorDataset(test_data, ['data'], shuffle=False).batch(TEST_BATCH_SIZE)

        model = CMMS_GCL()
        criterion = nn.BCEWithLogitsLoss()
        optimizer = nn.Adam(model.trainable_params(), learning_rate=LR)
        max_auc = 0

        model_file_name = 'model.ckpt'
        result_file_name = 'result.csv'
        for epoch in range(NUM_EPOCHS):
            train(model, train_loader, optimizer, epoch + 1)
            G, P = predicting(model, test_loader)

            auc, acc, precision, recall, f1_score, mcc = metric(G, P)
            ret = [auc, acc, precision, recall, f1_score, mcc]
            if auc > max_auc:
                max_auc = auc
                ms.save_checkpoint(model, model_file_name)
                with open(result_file_name, 'w') as f:
                    f.write(','.join(map(str, ret)))
            print(f'{auc:.4f}\t {acc:.4f}\t {precision:.4f}\t {recall:.4f}\t{f1_score:.4f}\t {mcc:.4f}')

        print('Maximum acc found. Model saved.')
