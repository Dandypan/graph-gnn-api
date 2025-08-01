import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, hidden_layers, activation, regularization):
        super(GCN, self).__init__()
        self.regularization = regularization
        self.activation = getattr(F, activation.lower())

        self.layers = torch.nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim))
        for _ in range(hidden_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))
        self.layers.append(GCNConv(hidden_dim, output_dim))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.layers[:-1]:
            x = self.activation(conv(x, edge_index))
        x = self.layers[-1](x, edge_index)
        return x


def train(G, features, labels, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = Data(x=features, edge_index=G.edge_index, y=labels).to(device)
    model = GCN(
        input_dim=features.shape[1],
        hidden_dim=16,
        output_dim=len(labels.unique()),
        hidden_layers=config.hidden_layers,
        activation=config.activation,
        regularization=config.regularization
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.regularization_rate)

    model.train()
    for epoch in range(config.epoch):
        optimizer.zero_grad()
        out = model(data)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    pred = model(data).argmax(dim=1)
    correct = int((pred[data.test_mask] == data.y[data.test_mask]).sum())
    acc = correct / int(data.test_mask.sum())

    return {
        "accuracy": acc.item(),
        "message": "GCN training complete"
    }
