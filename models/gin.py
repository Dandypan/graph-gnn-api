import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from utils.dataset import load_default_graph
from schemas import GNNParams

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
    def __init__(self, input_dim, hidden_dims, output_dim, activation):
        super(GIN, self).__init__()
        dims = [input_dim] + hidden_dims
        self.layers = nn.ModuleList(
            [GINLayer(dims[i], dims[i+1]) for i in range(len(dims)-1)]
        )
        self.output_layer = nn.Linear(dims[-1], output_dim)
        self.activation_fn = getattr(F, activation)

    def forward(self, x, adj):
        for layer in self.layers:
            x = layer(x, adj)
            x = self.activation_fn(x)
        return self.output_layer(x)

def run_gin(params: GNNParams):
    data = load_default_graph(params.noise_1, params.noise_2)
    x, adj, labels = data["x"], data["adj"], data["labels"]

    X_train, X_test, y_train, y_test = train_test_split(
        x, labels, train_size=params.train_test_ratio, stratify=labels
    )

    model = GIN(
        input_dim=x.shape[1],
        hidden_dims=params.hidden_layers,
        output_dim=len(torch.unique(labels)),
        activation=params.activation
    )

    criterion = nn.CrossEntropyLoss() if params.problem_type == "classification" else nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=params.learning_rate,
        weight_decay=params.regularization_rate if params.regularization == "l2" else 0.0
    )

    steps = [
        f"GIN model initialized with {len(params.hidden_layers)} hidden layers: {params.hidden_layers}",
        f"Activation function: {params.activation.upper()}",
        f"Training GIN for {params.epoch} epochs..."
    ]

    for epoch in range(params.epoch):
        model.train()
        optimizer.zero_grad()
        output = model(X_train, adj)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        pred_train = model(X_train, adj).argmax(dim=1)
        pred_test = model(X_test, adj).argmax(dim=1)
        acc_train = (pred_train == y_train).float().mean().item()
        acc_test = (pred_test == y_test).float().mean().item()

    steps += [
        f"Training complete. Accuracy: {acc_train:.2f}, Loss: {loss.item():.2f}",
        f"Evaluating on test set...",
        f"Test Accuracy: {acc_test:.2f}"
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
            "accuracy": acc_test,
            "loss": None
        },
        "steps": steps
    }
