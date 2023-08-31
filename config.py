from pathlib import Path 
import torch

def is_cuda_available():
   return torch.cuda.is_available()

def get_dst_device():
    return torch.device("cuda" if is_cuda_available() else "cpu")

"""
    1. num_epochs: Number of epochs to train the model
    2. d_model: Total dimensions to consider for input embeddings
    3. num_heads: Number of self attention heads
    4. num_layers: Number of Encoder / Decoder Blocks
    5. dropout: Regularization value
    6. d_ff: Expansion Coefficient for First linear layer in FeedForward Module
    7. lang_in: Input Language
    8. lang_out: Ouput Language (For Translation)
    9. lang_dataset: Dataset to be used for Training and Testing
    10. batch_size: Batch Size of Training & Testing
    11. device
"""

def get_config():
    return { 
            "batch_size": 6, 
            "num_epochs": 10, 
            "lr": 10**-4,
            "d_model": 512,
            "d_ff": 2048,
            "dropout": 0.1,
            "lang_dataset": "opus_books",
            "num_layers": 6,
            "num_heads": 8,
            "lang_in": "en", 
            "lang_out": "it", 
            "model_folder": "weights", 
            "model_basename": "tmodel_", 
            "preload": None, 
            "tokenizer_file": "tokenizer_{0}.json", 
            "experiment_name": "runs/tmodel",
            "device": get_dst_device()
            }
    
def get_weights_file_path(config, epoch:str): 
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.')/model_folder/model_filename)