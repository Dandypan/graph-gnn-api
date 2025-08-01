
from pydantic import BaseModel

class GNNConfig(BaseModel):
    algorithm: str
    learning_rate: float
    activation: str
    regularization: str
    regularization_rate: float
    problem_type: str
    epoch: int
    hidden_layers: int
    train_test_ratio: float
    noise_1: float
    noise_2: float
    batch_size: int
