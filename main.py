
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from schemas import GNNConfig
from gnn_runner import run_gnn_model

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/gnn/predict")
def predict(config: GNNConfig):
    return run_gnn_model(config)
