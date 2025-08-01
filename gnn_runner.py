from models.gcn import GCN
from utils.dataset import generate_synthetic_graph_data
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def run_gcn(req):

    lr = req.learning_rate
    activation = req.activation
    regularization = req.regularization
    reg_rate = req.regularization_rate
    batch_size = req.batch_size
    epochs = req.epoch
    task_type = req.problem_type
    hidden_layers = req.hidden_layers
    train_ratio = req.train_test_ratio
    noise_1 = req.noise_1
    noise_2 = req.noise_2


    data = generate_synthetic_graph_data(noise_level_1=noise_1, noise_level_2=noise_2)
    X, edge_index, y = data['features'], data['edge_index'], data['labels']
    num_nodes, input_dim = X.shape
    output_dim = 1 if task_type == "Regression" else len(torch.unique(y))

  
    split_idx = int(num_nodes * train_ratio)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

   
    model = GCN(
        input_dim=input_dim,
        hidden_dim=16,
        output_dim=output_dim,
        num_layers=hidden_layers,
        activation=activation
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)


    def regularization_loss():
        if regularization == "L2":
            return reg_rate * sum(torch.norm(p) for p in model.parameters())
        elif regularization == "L1":
            return reg_rate * sum(torch.norm(p, 1) for p in model.parameters())
        return 0


    history = []
    for epoch in range(epochs):
        model.train()
        out = model(X, edge_index)
        loss_fn = F.mse_loss if task_type == "Regression" else F.cross_entropy
        loss = loss_fn(out[:split_idx], y_train) + regularization_loss()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history.append({
            "step": epoch + 1,
            "info": f"Epoch {epoch+1} | Loss: {loss.item():.4f}"
        })

 
    model.eval()
    pred = model(X, edge_index)
    pred_labels = pred[split_idx:]
    y_true = y_test

    if task_type == "Classification":
        pred_cls = torch.argmax(pred_labels, dim=1)
        accuracy = (pred_cls == y_true).float().mean().item()
        result = {"accuracy": round(accuracy, 4)}
    else:
        mse = F.mse_loss(pred_labels, y_true).item()
        result = {"mse": round(mse, 4)}

    return history, result
