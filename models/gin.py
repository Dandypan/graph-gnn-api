import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.utils import to_dense_adj
from utils.dataset import load_default_graph
from schemas import GNNRequest

class GINMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GINMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class GINLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GINLayer, self).__init__()
        self.mlp = GINMLP(input_dim, hidden_dim, hidden_dim)
        self.eps = nn.Parameter(torch.zeros(1))

    def forward(self, x, adj):
        out = torch.matmul(adj, x) + (1 + self.eps) * x
        return self.mlp(out)

class GIN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation_fn):
        super(GIN, self).__init__()
        dims = [input_dim] + hidden_dims
        self.layers = nn.ModuleList(
            [GINLayer(dims[i], dims[i+1]) for i in range(len(dims) - 1)]
        )
        self.output_layer = nn.Linear(dims[-1], output_dim)
        self.activation_fn = activation_fn

    def forward(self, x, adj):
        for layer in self.layers:
            x = layer(x, adj)
            x = self.activation_fn(x)
        return self.output_layer(x)

def run_gin(params: GNNRequest):
    data = load_default_graph(params.noise_1, params.noise_2)
    x, edge_index, labels = data.x, data.edge_index, data.y
    adj = to_dense_adj(edge_index)[0]

    # Tensor-based train-test split
    num_nodes = x.size(0)
    train_size = int(num_nodes * params.train_test_ratio)
    indices = torch.randperm(num_nodes)
    train_idx, test_idx = indices[:train_size], indices[train_size:]

    x_train, x_test = x[train_idx], x[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

    # Activation selection
    activation_fn = {
        "relu": F.relu,
        "tanh": torch.tanh,
        "sigmoid": torch.sigmoid
    }.get(params.activation.lower(), F.relu)

    model = GIN(
        input_dim=x.shape[1],
        hidden_dims=params.hidden_layers,
        output_dim=len(torch.unique(labels)),
        activation_fn=activation_fn
    )

    criterion = nn.CrossEntropyLoss() if params.problem_type.lower() in ["classification", "node_classification"] else nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=params.learning_rate,
        weight_decay=params.regularization_rate if params.regularization == "l2" else 0.0
    )

    steps = [
        f"GIN initialized with {len(params.hidden_layers)} hidden layers: {params.hidden_layers}",
        f"Activation: {params.activation.upper()}, Epochs: {params.epoch}"
    ]

    # Training loop
    for epoch in range(params.epoch):
        model.train()
        optimizer.zero_grad()
        output = model(x_train, adj)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        pred_train = model(x_train, adj).argmax(dim=1)
        pred_test = model(x_test, adj).argmax(dim=1)
        acc_train = (pred_train == y_train).float().mean().item()
        acc_test = (pred_test == y_test).float().mean().item()

    steps += [
        f"Train Accuracy: {acc_train:.2f}, Final Loss: {loss.item():.4f}",
        f"Test Accuracy: {acc_test:.2f}"
    ]

    # Return predictions
    predictions = [
        {"node_id": int(i), "predicted_label": int(label)}
        for i, label in zip(test_idx.tolist(), pred_test.tolist())
    ]

    return {
        "summary": {
            "model": "GIN",
            "layers": [x.shape[1]] + params.hidden_layers + [len(torch.unique(labels))],
            "activation": params.activation.upper(),
            "epochs": params.epoch,
            "batch_size": params.batch_size
        },
        "train_metrics": {
            "accuracy": acc_train,
            "loss": loss.item()
        },
        "test_metrics": {
            "accuracy": acc_test
        },
        "predictions": predictions,
        "steps": steps
    }
