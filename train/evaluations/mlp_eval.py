import sys
import os
import torch
import torch.nn.functional as F
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..3")))

from retrieval.embedder import MLP

# Load MLP model and evaluate it on test query embeddings

def load_model(model_path, input_dim, hidden_dim, output_dim, device):
    model = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    model.load_state_dict(torch.load(model_path))
    model.to(device).eval()
    return model

def evaluate_model(model, test_query_embeddings, device):
    with torch.no_grad():
        test_query_embeddings = test_query_embeddings.to(device)
        output_embeddings = model(test_query_embeddings)
    return output_embeddings

if __name__ == "__main__":
    model_path = "models/MLPEmbedder_finetune2.pth"
    test_query_embeddings_path = "models/embeddings/test_embeddings_query.pt"
    test_code_embeddings_path = "models/embeddings/test_embeddings_code.pt"
    
    input_dim = 768
    hidden_dim = 512
    output_dim = 768
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the stored MLP model
    model = load_model(model_path, input_dim, hidden_dim, output_dim, device)
    
    # Load the test query and code embeddings
    test_query_embeddings = torch.load(test_query_embeddings_path).to(device)
    test_code_embeddings = torch.load(test_code_embeddings_path).to(device)
    
    # Evaluate the model on the test query embeddings
    transformed_query_embeddings = evaluate_model(model, test_query_embeddings, device)
    torch.save(transformed_query_embeddings, "models/embeddings/test_embeddings_query_mlp.pt")
    
    # Calculate the cosine similarity between the transformed query embeddings and the code embeddings
    cosine_similarities = F.cosine_similarity(transformed_query_embeddings, test_code_embeddings, dim=1)
    
    np.save("train/results/mlp_cosine_similarities.npy", cosine_similarities.cpu().numpy())