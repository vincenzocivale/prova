import os
import pandas as pd
import numpy as np
from datasets import Dataset
from pathlib import Path
import torch
import scanpy as sc
from tqdm import tqdm

def load_h5ad_file(file_path):
    """
    Loads an .h5ad file, extracts gene IDs, sequence data, and labels.
    Handles sparse matrices by converting them to dense NumPy arrays.
    
    Args:
        file_path (str): Path to the .h5ad file.
        
    Returns:
        tuple: (gene_ids, seq_data, labels)
               - gene_ids (np.ndarray): Gene names.
               - seq_data (np.ndarray): Gene expression data (n_cells x n_genes).
               - labels (np.ndarray): Labels (corresponding to seq_data in this context).
    """
    adata = sc.read_h5ad(file_path)
    gene_ids = adata.var_names.to_numpy()
    
    # Convert seq_data to numpy array, handling sparse matrix
    seq_data = adata.X
    if not isinstance(seq_data, np.ndarray):
        seq_data = np.vstack([row.toarray().squeeze() for row in seq_data])
    
    # In this context, labels are the gene expression values themselves
    labels = seq_data 
    
    return gene_ids, seq_data, labels

def inizialize_record_dict(model):
    """
    Initializes a dictionary to store transformer input embeddings.
    
    Args:
        model (torch.nn.Module): The PyTorch model containing a 'transformer' attribute.
                                 
    Returns:
        dict: Dictionary with 'label' and 'transformer_input_embedding' keys.
    """
    layer_embeddings = {'label': []}
    # We will now capture the input to the entire transformer block
    layer_embeddings['transformer_input_embedding'] = [] 
    return layer_embeddings

def create_dataset(data_fold_path, tokenizer, model):
    """
    Creates a dataset by extracting the intermediate embeddings from each transformer layer
    of a model for gene expression data.
    
    Args:
        data_fold_path (str): Path to the folder containing .h5ad files.
        tokenizer: The tokenizer object to convert cell vectors into tokens.
        model (torch.nn.Module): The PyTorch model from which to extract embeddings.
        
    Returns:
        datasets.Dataset: A Dataset object containing labels, file origin,
                          and the intermediate embeddings from each transformer layer.
    """
    # Initialize device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()  # Set model to evaluation mode (disables dropout, etc.)

    # Initialize the dictionary to store final data
    record_dict = {
        'label': [],
        'file_origin': [],
        'transformer_input_embedding': [],  # Input to transformer (layer 0)
    }
    
    # Add keys for each transformer layer output
    num_layers = len(model.transformer.layers)
    for i in range(num_layers):
        record_dict[f'transformer_layer_{i}_output'] = []
    
    # Find all .h5ad or .h5ad.gz files in the specified folder
    files = [f for f in os.listdir(data_fold_path) if f.endswith(".h5ad.gz") or f.endswith(".h5ad")]
    print(f"Found {len(files)} files to process.")
    
    # Progress bar for files
    progress_bar_files = tqdm(files, desc="Processing files", unit="file")
    
    for file in progress_bar_files:
        file_path = os.path.join(data_fold_path, file)
        gene_ids, seq_data, labels = load_h5ad_file(file_path)
        
        # Tokenize cell data
        tokenized_data = tokenizer.tokenize_cell_vectors(seq_data, gene_ids)
        
        # Progress bar for cells within each file
        progress_bar_cells = tqdm(enumerate(tokenized_data), total=len(tokenized_data), 

        
        # Process one cell at a time
        for cell_idx, (cell_tokens, cell_values) in progress_bar_cells:
            # Dictionary to store intermediate embeddings for the current cell
            intermediate_embeddings = {}
            
            # Hook function to capture transformer input
            def get_transformer_input_hook():
                def pre_hook(module, input):
                    transformer_input_tensor = input[0]
                    intermediate_embeddings['transformer_input'] = transformer_input_tensor.detach().cpu()
                    return input
                return pre_hook
            
            # Hook function to capture each layer's output
            def get_layer_output_hook(layer_idx):
                def forward_hook(module, input, output):
                    # output is the hidden states after this layer
                    intermediate_embeddings[f'layer_{layer_idx}'] = output.detach().cpu()
                return forward_hook
            
            # Register hooks
            hooks = []
            
            # Hook for transformer input
            transformer_input_hook = model.transformer.register_forward_pre_hook(get_transformer_input_hook())
            hooks.append(transformer_input_hook)
            
            # Hooks for each transformer layer output
            for i, layer in enumerate(model.transformer.layers):
                layer_hook = layer.register_forward_hook(get_layer_output_hook(i))
                hooks.append(layer_hook)
            
            # Move tensors to device (GPU/CPU)
            cell_tokens = cell_tokens.to(device)
            cell_values = cell_values.to(device)
            
            # Create attention mask (assuming 0 is the padding value)
            attention_mask = (cell_values != 0).to(device)
            
            with torch.no_grad():
                # Perform the model's forward pass. The hooks will populate intermediate_embeddings.
                _ = model(cell_tokens.unsqueeze(0), cell_values.unsqueeze(0), 
                         attention_mask=attention_mask.unsqueeze(0))
            
            # Process and store all intermediate embeddings
            record_dict['label'].append(labels[cell_idx].tolist())
            record_dict['file_origin'].append(file)
            
            # Store transformer input embedding
            if 'transformer_input' in intermediate_embeddings:
                input_embedding = intermediate_embeddings['transformer_input']
                # Handle batch dimension removal more safely
                if input_embedding.dim() > 2:
                    processed_input_embedding = input_embedding[0].numpy()  # Take first batch item
                else:
                    processed_input_embedding = input_embedding.numpy()
                
                # Remove CLS token if present (assuming first token is CLS)
                if processed_input_embedding.shape[0] == 5001:
                    processed_input_embedding = processed_input_embedding[1:]
                
                record_dict['transformer_input_embedding'].append(processed_input_embedding.tolist())
            else:
                print(f"Warning: No transformer input captured for cell {cell_idx} in file {file}")
                record_dict['transformer_input_embedding'].append([])
            
            # Store each layer's output embedding
            for i in range(num_layers):
                layer_key = f'layer_{i}'
                if layer_key in intermediate_embeddings:
                    layer_embedding = intermediate_embeddings[layer_key]
                    # Handle batch dimension removal more safely
                    if layer_embedding.dim() > 2:
                        processed_layer_embedding = layer_embedding[0].numpy()  # Take first batch item
                    else:
                        processed_layer_embedding = layer_embedding.numpy()
                    
                    # Remove CLS token if present
                    if processed_layer_embedding.shape[0] == 5001:
                        processed_layer_embedding = processed_layer_embedding[1:]
                    
                    record_dict[f'transformer_layer_{i}_output'].append(processed_layer_embedding.tolist())
                else:
                    print(f"Warning: No layer {i} output captured for cell {cell_idx} in file {file}")
                    record_dict[f'transformer_layer_{i}_output'].append([])
            
            # Remove all hooks after processing the current cell
            for hook in hooks:
                hook.remove()
    
    # Create a Dataset object from the final dictionary
    dataset = Dataset.from_dict(record_dict)
    return dataset