import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from retrieval.embedder import MLP

# Load MLP model and evaluate it on test query embeddings

def load_model(model_path, input_dim, hidden_dim, output_dim, device, mapping_block, residual, n_blocks):
    if mapping_block == 'MLP':
        model = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, residual=residual)
    elif mapping_block == 'FFN':
        from retrieval.embedder import FFN
        model = FFN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, residual=residual, num_layers=n_blocks)
    model.load_state_dict(torch.load(model_path))
    model.to(device).eval()
    return model

def evaluate_model(model, test_query_embeddings, device):
    with torch.no_grad():
        test_query_embeddings = test_query_embeddings.to(device)
        output_embeddings = model(test_query_embeddings)
    return output_embeddings

def main(args):
    print('args:', args)
    if args.device == "cuda":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {args.device}")
    
    input_dim = 768
    hidden_dim = 512
    output_dim = 768
    
    model = load_model(args.model_path, input_dim, hidden_dim, output_dim, args.device, args.mapping_block, args.residual, args.n_blocks)
    
    # Load the test query and code embeddings
    test_query_embeddings = torch.load(args.test_query_embeddings_path).to(args.device)
    test_code_embeddings = torch.load(args.test_code_embeddings_path).to(args.device)
    
    transformed_query_embeddings = evaluate_model(model, test_query_embeddings, args.device)
    
    torch.save(transformed_query_embeddings, args.transformed_query_embeddings_path)
    
    # Calculate the cosine similarity between the transformed query embeddings and the code embeddings
    cosine_similarities = F.cosine_similarity(transformed_query_embeddings, test_code_embeddings, dim=1)
    
    similarity_path = "models/embeddings/"+args.transformed_query_embeddings_path.split('/')[-1].split('.')[0] + '_cosine_similarities.npy'
    np.save(similarity_path, cosine_similarities.cpu().numpy())
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected')
    parser.add_argument("--mapping_block", type=str, default='FFN', help="Mapping block to use (MLP or Linear or FFN)")
    parser.add_argument("--model_path", type=str, help="Path to the stored MLP model", default="models/MLPEmbedder_finetune2.pth")
    parser.add_argument("--test_query_embeddings_path", type=str, help="Path to the test query embeddings", default="models/embeddings/test_embeddings_query.pt")
    parser.add_argument("--test_code_embeddings_path", type=str, help="Path to the test code embeddings", default="models/embeddings/test_embeddings_code.pt")
    parser.add_argument("--transformed_query_embeddings_path", type=str, help="Path to save the transformed query embeddings", default="models/embeddings/test_embeddings_query_mlp.pt")
    parser.add_argument("--device", type=str, help="Device to use (cpu or cuda)", default="cuda")
    parser.add_argument("--cosine_similarities_path", type=str, help="Path to save the cosine similarities", default="models/embeddings/cosine_similarities.npy")
    parser.add_argument("--residual", type=str2bool, default=False, help="Whether to use residual connection in the mapping block")
    parser.add_argument("--n_blocks", type=int, help="Number of FFN blocks")
    args = parser.parse_args()
    
    main(args)