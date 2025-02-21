# TODO: Implement this module
import random
import re
import faiss
import numpy as np

class NegativeExampleGenerator:
    def __init__(self):
        pass
    
    def get_random_negative(self, queries, positives):
        """
        Selects a random non-matching code snippet as a negative.
        """
        neg_candidates = positives.copy()
        random.shuffle(neg_candidates)
        
        negatives = []
        for q, p in zip(queries, positives):
            neg = random.choice([n for n in neg_candidates if n != p])
            negatives.append(neg)
        
        return negatives

    def get_semantic_hard_negatives(self, queries, positives, code_snippets, model, top_k=5):
        """
        Finds semantically similar but incorrect negatives using embeddings.
        """
        embeddings = np.array([model.encode(code) for code in code_snippets])
        
        # Build FAISS index
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

        negatives = []
        for query in queries:
            query_embedding = np.array([model.encode(query)])
            _, indices = index.search(query_embedding, top_k)  # Retrieve closest matches
            
            for idx in indices[0]:
                if code_snippets[idx] not in positives:
                    negatives.append(code_snippets[idx])
                    break
            else:
                negatives.append(random.choice(code_snippets))  # Fallback to random

        return negatives
    
    def generate_adversarial_negative(self, code_snippet):
        """
        Creates an adversarial negative example by modifying function names.
        """
        adversarial_code = re.sub(r'\bdef (\w+)', r'def \1_modified', code_snippet)
        adversarial_code = re.sub(r'\b(\w+) = ', r'\1_alt = ', adversarial_code)
        return adversarial_code
