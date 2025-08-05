from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from schemas import GNNRequest
from gnn_runner import train_and_predict

app = FastAPI(title="Graph GNN API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/gnn/predict")
async def predict(request: GNNRequest):
    result = run_gnn(request)
    return result
