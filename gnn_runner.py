import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from models.gcn import GCN
from models.gin import GIN
from utils.dataset import load_default_graph


def get_activation_fn(name):
    return {
        "relu": F.relu,
        "tanh": torch.tanh,
        "sigmoid": torch.sigmoid
    }.get(name, F.relu)


def apply_regularization(model, reg_type, reg_rate):
    reg_loss = 0.0
    if reg_type == "L1":
        for param in model.parameters():
            reg_loss += torch.sum(torch.abs(param))
    elif reg_type == "L2":
        for param in model.parameters():
            reg_loss += torch.sum(param ** 2)
    return reg_rate * reg_loss


def train_and_predict(request):
    # 1. Load and prepare data
    data = load_default_graph(noise_1=request.noise_1, noise_2=request.noise_2)
    num_nodes = data.num_nodes
    input_dim = data.x.size(1)
    output_dim = torch.max(data.y).item() + 1

    # 2. Split dataset
    train_size = int(num_nodes * request.train_test_ratio)
    test_size = num_nodes - train_size
    indices = torch.randperm(num_nodes)
    train_idx, test_idx = indices[:train_size], indices[train_size:]

    # 3. Initialize model
    activation_fn = get_activation_fn(request.activation)
    if request.algorithm == "GCN":
        model = GCN(input_dim, request.hidden_layers, output_dim, activation_fn)
    elif request.algorithm == "GIN":
        model = GIN(input_dim, request.hidden_layers, output_dim, activation_fn)
    else:
        raise ValueError("Unsupported GNN algorithm")

    optimizer = torch.optim.Adam(model.parameters(), lr=request.learning_rate)

    # 4. Training loop
    logs = []
    for epoch in range(1, request.epoch + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[train_idx], data.y[train_idx])
        if request.regularization != "None":
            loss += apply_regularization(model, request.regularization, request.regularization_rate)
        loss.backward()
        optimizer.step()

        acc = (out[test_idx].argmax(dim=1) == data.y[test_idx]).float().mean().item()
        logs.append({
            "step": epoch,
            "description": f"Epoch {epoch}",
            "log": {
                "loss": round(loss.item(), 4),
                "accuracy": round(acc, 4)
            }
        })

    # 5. Final predictions
    model.eval()
    final_out = model(data.x, data.edge_index)
    probs = F.softmax(final_out, dim=1)
    predicted_labels = torch.argmax(probs, dim=1)

    predictions = []
    for i in range(num_nodes):
        predictions.append({
            "node_id": i,
            "label": predicted_labels[i].item(),
            "probs": [round(p.item(), 4) for p in probs[i]]
        })

    final_acc = (predicted_labels[test_idx] == data.y[test_idx]).float().mean().item()
    summary = {
        "final_accuracy": round(final_acc, 4),
        "final_loss": round(loss.item(), 4),
        "model": request.algorithm,
        "epochs": request.epoch
    }

    return {
        "steps": logs,
        "summary": summary,
        "predictions": predictions
    }
