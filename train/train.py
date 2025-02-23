import sys
import os
import torch
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses
)
from sentence_transformers.training_args import BatchSamplers
from datasets import Dataset
# from torch.utils.data import DataLoader
# from train.negative_example_generator import NegativeExampleGenerator
from dataset_python import PythonDataset
from tools.logger import logger


def fine_tune_sentence_bert(model_name="jinaai/jina-embeddings-v2-base-code", train_dataset=None, use_hard_negatives=True, train_epochs=3, batch_size=8, save_path="models/fine_tuned_embedder"):
    """
    Fine-tune the SentenceTransformer model using contrastive learning.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, trust_remote_code=True).to(device)

    # Define Loss functions
    loss = losses.MultipleNegativesRankingLoss(model)

    # Convert train_dataset to Dataset object if it's not already
    if isinstance(train_dataset, PythonDataset):
        train_dataset = train_dataset.to_hf_dataset()

    # Train model
    logger.info("Starting fine-tuning...")
    args = SentenceTransformerTrainingArguments(
        output_dir="models/",
        num_train_epochs=train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        fp16=True,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
    )
    
    # Create a trainer & train
    logger.info("Starting fine-tuning with contrastive learning...")
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
    )
    trainer.train()
    
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    logger.info(f"Fine-tuned model saved at: {save_path}")

def train_embedder(model, train_dataset, train_epochs=3, batch_size=8, save_path="models/fine_tuned_jina_embeddings"):
    """
    Fine-tune the embedder using contrastive learning.
    """
    
    '''
    --- Original code ---
    --- Now we should modify dataset to switch between hard negatives and random negatives ---
    # Extract positive examples from train_dataset
    queries = train_dataset["query"]
    positives = train_dataset["positive"]

    # Generate negative examples
    negative_generator = NegativeExampleGenerator()
    if use_hard_negatives:
        negatives = negative_generator.get_semantic_hard_negatives(queries, positives, positives + queries, model)
    else:
        negatives = negative_generator.get_random_negative(queries, positives)

    # Convert data into a Hugging Face Dataset (WITHOUT using InputExample)
    train_dataset = Dataset.from_dict({
        "anchor": queries,  # Renamed to match SentenceTransformer's triplet format
        "positive": positives,
        "negative": negatives
    })
    '''
    # Loss functions:
    loss = losses.MultipleNegativesRankingLoss(model)
    
    args = SentenceTransformerTrainingArguments(
        output_dir="models/",
        num_train_epochs=train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        fp16=True,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
    )
    
    # Create a trainer & train
    logger.info("Starting fine-tuning with contrastive learning...")
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
    )
    trainer.train()

    model.save_pretrained(save_path)
    logger.info(f"Fine-tuned model saved at: {save_path}")

# Example Usage
if __name__ == "__main__":
    train_data = [
    {
        "query": "What is the summation function?",
        "positive": "def add(x, y):\n  return x + y",
        "negative": "def Multiply(num1, num2):\n    answer = num1 * num2\n    return answer"
    },
    {
        "query": "How to find the maximum of two numbers?",
        "positive": "def max_value(a, b):\n  return a if a > b else b",
        "negative": "def min_value(a, b):\n  return a if a < b else b"
    },
    {
        "query": "Function to calculate the factorial of a number?",
        "positive": "def factorial(n):\n  return 1 if n == 0 else n * factorial(n - 1)",
        "negative": "def fibonacci(n):\n  return n if n <= 1 else fibonacci(n - 1) + fibonacci(n - 2)"
    },
    {
        "query": "How to check if a number is even?",
        "positive": "def is_even(n):\n  return n % 2 == 0",
        "negative": "def is_odd(n):\n  return n % 2 != 0"
    },
    {
        "query": "Function to reverse a string?",
        "positive": "def reverse_string(s):\n  return s[::-1]",
        "negative": "def capitalize_string(s):\n  return s.capitalize()"
    },
    {
        "query": "How to calculate the square of a number?",
        "positive": "def square(n):\n  return n * n",
        "negative": "def cube(n):\n  return n * n * n"
    },
    {
        "query": "Function to check if a string is a palindrome?",
        "positive": "def is_palindrome(s):\n  return s == s[::-1]",
        "negative": "def reverse_words(s):\n  return ' '.join(s.split()[::-1])"
    },
    {
        "query": "How to find the greatest common divisor (GCD) of two numbers?",
        "positive": "def gcd(a, b):\n  while b:\n    a, b = b, a % b\n  return a",
        "negative": "def lcm(a, b):\n  return (a * b) // gcd(a, b)"
    },
    {
        "query": "Function to sort a list in ascending order?",
        "positive": "def sort_list(lst):\n  return sorted(lst)",
        "negative": "def reverse_list(lst):\n  return lst[::-1]"
    },
    {
        "query": "How to convert a string to lowercase?",
        "positive": "def to_lower(s):\n  return s.lower()",
        "negative": "def to_upper(s):\n  return s.upper()"
    },
    {
        "query": "How to find the length of a string?",
        "positive": "def string_length(s):\n  return len(s)",
        "negative": "def count_vowels(s):\n  return sum(1 for c in s if c in 'aeiouAEIOU')"
    },
    {
        "query": "How to check if a number is prime?",
        "positive": "def is_prime(n):\n  if n < 2:\n    return False\n  for i in range(2, int(n ** 0.5) + 1):\n    if n % i == 0:\n      return False\n  return True",
        "negative": "def is_composite(n):\n  return not is_prime(n)"
    },
    {
        "query": "Function to compute the nth Fibonacci number?",
        "positive": "def fibonacci(n):\n  return n if n <= 1 else fibonacci(n - 1) + fibonacci(n - 2)",
        "negative": "def factorial(n):\n  return 1 if n == 0 else n * factorial(n - 1)"
    },
    {
        "query": "How to calculate the area of a circle?",
        "positive": "def circle_area(radius):\n  return 3.14159 * radius * radius",
        "negative": "def circle_circumference(radius):\n  return 2 * 3.14159 * radius"
    },
    {
        "query": "Function to check if a number is positive?",
        "positive": "def is_positive(n):\n  return n > 0",
        "negative": "def is_negative(n):\n  return n < 0"
    },
    {
        "query": "How to merge two lists?",
        "positive": "def merge_lists(list1, list2):\n  return list1 + list2",
        "negative": "def intersect_lists(list1, list2):\n  return [x for x in list1 if x in list2]"
    },
    {
        "query": "How to count occurrences of an element in a list?",
        "positive": "def count_occurrences(lst, item):\n  return lst.count(item)",
        "negative": "def remove_duplicates(lst):\n  return list(set(lst))"
    },
    {
        "query": "How to find the average of a list of numbers?",
        "positive": "def average(lst):\n  return sum(lst) / len(lst) if lst else 0",
        "negative": "def median(lst):\n  sorted_lst = sorted(lst)\n  n = len(sorted_lst)\n  return (sorted_lst[n//2] if n % 2 != 0 else (sorted_lst[n//2 - 1] + sorted_lst[n//2]) / 2)"
    },
    {
        "query": "Function to convert Celsius to Fahrenheit?",
        "positive": "def celsius_to_fahrenheit(c):\n  return (c * 9/5) + 32",
        "negative": "def fahrenheit_to_celsius(f):\n  return (f - 32) * 5/9"
    },
    {
        "query": "How to check if a string contains only digits?",
        "positive": "def is_numeric(s):\n  return s.isdigit()",
        "negative": "def contains_alphabet(s):\n  return any(c.isalpha() for c in s)"
    }
]
    
    train_dataset = PythonDataset("tests/data/python", seed=224, negative_triplets=True, negative_precomputed=True)
    fine_tune_sentence_bert(train_dataset=train_dataset)
