import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import NNConv


class DGN(torch.nn.Module):
    def __init__(self, MODEL_PARAMS):
        super(DGN, self).__init__()
        self.model_params = MODEL_PARAMS
        
        nn = Sequential(Linear(self.model_params["Linear1"]["in"], self.model_params["Linear1"]["out"]), ReLU()) # in=6, out=36
        self.conv1 = NNConv(self.model_params["conv1"]["in"], self.model_params["conv1"]["out"], nn, aggr='mean') # in=1, out=36
        
        nn = Sequential(Linear(self.model_params["Linear2"]["in"], self.model_params["Linear2"]["out"]), ReLU())
        self.conv2 = NNConv(self.model_params["conv2"]["in"], self.model_params["conv2"]["out"], nn, aggr='mean')
        
        nn = Sequential(Linear(self.model_params["Linear3"]["in"], self.model_params["Linear3"]["out"]), ReLU())
        self.conv3 = NNConv(self.model_params["conv3"]["in"], self.model_params["conv3"]["out"], nn, aggr='mean')
        
        
    def forward(self, data):
        """
            Args:
                data (Object): data object consist of three parts x, edge_attr, and edge_index.
                                This object can be produced by using helper.cast_data function
                        x: Node features with shape [number_of_nodes, 1] (Simply set to vector of ones since we dont have any)
                        edge_attr: Edge features with shape [number_of_edges, number_of_views]
                        edge_index: Graph connectivities with shape [2, number_of_edges] (COO format) 
                        

        """
        x, edge_attr, edge_index = data.x, data.edge_attr, data.edge_index
        
        x = F.relu(self.conv1(x, edge_index, edge_attr))

        x = F.relu(self.conv2(x, edge_index, edge_attr))
        
        x = F.relu(self.conv3(x, edge_index, edge_attr))
        repeated_out = x.repeat(self.model_params["N_ROIs"],1,1)
        repeated_t   =  torch.transpose(repeated_out, 0, 1)
        diff = torch.abs(repeated_out - repeated_t)
        cbt = torch.sum(diff, 2)
        
        return cbt