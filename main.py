from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from schemas import GNNRequest
from gnn_runner import run_gnn

app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/gnn/predict")
def predict_gnn(req: GNNRequest):
    steps, result = run_gnn(req)
    return {"steps": steps, "result": result}
