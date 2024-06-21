from mindspore import Tensor 
import mindspore.dataset as ds 
import mindspore.dataset.transforms.c_transforms as C 
import os

class TestbedDataset: 
    def init(self, root='/data', dataset=None, xd=None, y=None, transform=None, pre_transform=None, smile_graph=None): #root is required for save preprocessed data, default is ‘/tmp’ 
        super(TestbedDataset, self).init(root, transform, pre_transform) # benchmark dataset, default = ‘davis’ 
        self.dataset = dataset 
        if os.path.isfile(self.processed_paths[0]): 
            print('Pre-processed data found: {}, loading …'.format(self.processed_paths[0])) 
            self.data, self.slices = ds.load(self.processed_paths[0]) 
        else: 
            print('Pre-processed data {} not found, doing pre-processing…'.format(self.processed_paths[0])) 
            self.process(xd, y, smile_graph) 
            self.data, self.slices = ds.load(self.processed_paths[0])

@property
def raw_file_names(self):
    pass
    #return ['some_file_1', 'some_file_2', ...]

@property
def processed_file_names(self):
    return [self.dataset + '.pt']

def download(self):
    # Download to `self.raw_dir`
    pass

def _download(self):
    pass

def _process(self):
    if not os.path.exists(self.processed_dir):
        os.makedirs(self.processed_dir)

def process(self, xd, y, smile_graph):
    assert (len(xd) == len(y)), "The two lists must be the same length!"
    data_list = []
    data_len = len(xd)
    for i in range(data_len):
        print('Converting SMILES to graph: {}/{}'.format(i+1, data_len))
        smiles = xd[i]
        labels = y[i]
        c_size, features, edge_index, smi_em = smile_graph[smiles]
        GCNData = data.Data(x=Tensor(features),
                            edge_index=Tensor(edge_index).transpose(1, 0),
                            smi_em=Tensor(smi_em),
                            y=Tensor([labels]))
        GCNData.__setitem__('c_size', Tensor([c_size]))
        data_list.append(GCNData)

    if self.pre_filter is not None:
        data_list = [data for data in data_list if self.pre_filter(data)]

    if self.pre_transform is not None:
        data_list = [self.pre_transform(data) for data in data_list]
    print('Graph construction done. Saving to file.')
    data, slices = self.collate(data_list)
    # save preprocessed data:
    ds.save((data, slices), self.processed_paths[0])