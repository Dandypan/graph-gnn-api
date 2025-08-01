
from models import gcn, gin
from utils.dataset import generate_graph

def run_gnn_model(config):
    G, features, labels = generate_graph(
        noise1=config.noise_1, noise2=config.noise_2,
        batch_size=config.batch_size
    )
    if config.algorithm.upper() == "GCN":
        output = gcn.train(G, features, labels, config)
    elif config.algorithm.upper() == "GIN":
        output = gin.train(G, features, labels, config)
    else:
        return {"error": "Unsupported model"}

    return {"status": "success", "log": output}
