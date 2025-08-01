GraphLab GNN API

This is the backend for the GraphLab GNN visualization and prediction system. It supports two GNN algorithms (GCN and GIN) and classical algorithms like Hopcroft-Karp for maximum bipartite matching. The system is designed for frontend compatibility and interactive experiments.

⸻

🔧 Setup

1. Clone the repo

git clone https://github.com/Dandypan/graph-gnn-api.git
cd graph-gnn-api

2. Install dependencies

pip install -r requirements.txt

3. Run FastAPI server

uvicorn main:app --host 0.0.0.0 --port 8000 --reload


⸻

📌 Available Endpoints

POST /gnn/predict

Run GCN or GIN model on a locked complex graph.

Request Body Example:

{
  "algorithm": "GCN",
  "learning_rate": 0.01,
  "activation": "relu",
  "regularization": "l2",
  "regularization_rate": 0.001,
  "problem_type": "node_classification",
  "epoch": 100,
  "hidden_layers": [32, 16],
  "train_test_ratio": 0.8,
  "noise_1": 0.1,
  "noise_2": 0.05,
  "batch_size": 16
}

Response Example:

{
  "status": "success",
  "algorithm": "GCN",
  "graph_id": "large-bipartite",
