from pydantic import BaseModel
from typing import Literal

class GNNRequest(BaseModel):
    algorithm: Literal['GCN', 'GIN']
    learning_rate: float
    activation: Literal['ReLU', 'Sigmoid', 'Tanh']
    regularization: Literal['L1', 'L2', 'None']
    regularization_rate: float
    problem_type: Literal['Classification', 'Regression']
    epoch: int
    hidden_layers: int
    train_test_ratio: float  # 0.0 ~ 1.0
    noise_1: float
    noise_2: float
    batch_size: int
