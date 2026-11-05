import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from .utils import location_to_edge, cross_dist
from torch_geometric.utils import k_hop_subgraph

class spatialDataset(Dataset):
    
    def __init__(self, data_fit, data_supervision, location, return_index=False):
        assert (data_supervision is None or len(data_fit) == len(data_supervision)) and len(data_fit) == len(location)
        self.data_fit = data_fit
        self.data_supervision = data_supervision
        self.location = location
        self.return_index = return_index
    
    def __len__(self):
        return len(self.data_fit)
    
    def __getitem__(self, index):
        if self.data_supervision is None:
            item = (self.data_fit[index], None, self.location[index])
        else:
            item = (self.data_fit[index], self.data_supervision[index], self.location[index])

        if self.return_index:
            return (*item, index)
        return item

class rnaDataset(Dataset):
    
    def __init__(self, data_fit, data_supervision, label):
        self.data_fit = data_fit
        self.data_supervision = data_supervision
        self.label = label
    
    def __len__(self):
        return len(self.data_fit)
    
    def __getitem__(self, index):
        return self.data_fit[index], self.data_supervision[index], self.label[index]

class spatialCollate():
    
    def __init__(self, knn = 10, device='cpu'):
        self.device = device
        self.knn = knn
    
    def __call__(self, batch):
        batch_fit, batch_supervision, batch_location = zip(*batch)
        batch_fit = torch.from_numpy(np.vstack(batch_fit)).to(self.device)
        if batch_supervision[0] is not None:
            batch_supervision = torch.from_numpy(np.vstack(batch_supervision)).to(self.device)
        batch_location = np.vstack(batch_location)
        batch_edge = location_to_edge(batch_location, self.knn).to(self.device)
        
        return batch_fit, batch_supervision, batch_edge


class globalSpatialCollate():
    """
    Build a training mini-batch from the GLOBAL spatial KNN graph.

    The DataLoader batch contains the seed ST cells. Because stAI has one GAT
    layer, we add the 1-hop incoming neighbors of those seeds from the global
    graph.  The returned seed_local_idx identifies the original seed cells
    inside the expanded local graph; stAI uses only those seed cells for losses.
    """

    def __init__(self, dataset, global_edge_index, device='cpu'):
        self.dataset = dataset
        self.global_edge_index = global_edge_index.cpu().long()
        self.device = device
        self.num_nodes = len(dataset)

    def __call__(self, batch):
        # batch item = (fit, supervision, location, global_index)
        seed_global_idx = torch.as_tensor(
            [int(item[3]) for item in batch], dtype=torch.long
        )

        # For the one-layer GAT, collect all source nodes that can send a
        # message to the seed nodes in the full graph.
        subset, local_edge_index, seed_local_idx, _ = k_hop_subgraph(
            node_idx=seed_global_idx,
            num_hops=1,
            edge_index=self.global_edge_index,
            relabel_nodes=True,
            num_nodes=self.num_nodes,
            flow='source_to_target',
        )

        subset_np = subset.cpu().numpy()

        expanded_fit = np.asarray(self.dataset.data_fit[subset_np], dtype=np.float32)
        expanded_fit = torch.from_numpy(expanded_fit).to(self.device)

        if self.dataset.data_supervision is None:
            expanded_supervision = None
        else:
            expanded_supervision = np.asarray(
                self.dataset.data_supervision[subset_np], dtype=np.float32
            )
            expanded_supervision = torch.from_numpy(expanded_supervision).to(self.device)

        return (
            expanded_fit,
            expanded_supervision,
            local_edge_index.to(self.device),
            seed_local_idx.to(self.device),
        )


class rnaCollate():
    
    def __init__(self, device='cpu'):
        self.device = device
    
    def __call__(self, batch):
        batch_fit, batch_supervision, batch_label = zip(*batch)
        batch_fit = torch.from_numpy(np.vstack(batch_fit)).to(self.device)
        batch_supervision = torch.from_numpy(np.vstack(batch_supervision)).to(self.device)
        batch_label = torch.from_numpy(np.vstack(batch_label)).long().squeeze(-1).to(self.device)
        batch_genegraph = cross_dist(batch_fit, batch_supervision)
        
        return batch_fit, batch_supervision, batch_label, batch_genegraph