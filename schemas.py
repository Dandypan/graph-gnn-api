from pydantic import BaseModel
from typing import List, Literal, Optional, Dict, Any

class GNNRequest(BaseModel):
    algorithm: Literal["GCN", "GIN"]
    learning_rate: float = 0.01
    activation: Literal["relu", "tanh", "sigmoid"] = "relu"
    regularization: Optional[Literal["L1", "L2", "None"]] = "None"
    regularization_rate: Optional[float] = 0.0
    problem_type: Literal["node_classification", "link_prediction"] = "node_classification"
    epoch: int = 100
    hidden_layers: List[int] = [64, 32]
    train_test_ratio: float = 0.8
    noise_1: float = 0.0
    noise_2: float = 0.0
    batch_size: int = 32

class PredictionStep(BaseModel):
    step: int
    description: str
    log: Dict[str, Any]

class GNNResponse(BaseModel):
    steps: List[PredictionStep]
    summary: Dict[str, Any]
