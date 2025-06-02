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
        seq_data = seq_data.toarray()
    
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
    Creates a dataset by extracting the input embeddings to the transformer block
    of a model for gene expression data.
    
    Args:
        data_fold_path (str): Path to the folder containing .h5ad files.
        tokenizer: The tokenizer object to convert cell vectors into tokens.
        model (torch.nn.Module): The PyTorch model from which to extract embeddings.
        
    Returns:
        datasets.Dataset: A Dataset object containing labels, file origin,
                          and the input embeddings for the transformer.
    """
    # Initialize device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()  # Set model to evaluation mode (disables dropout, etc.)

    # Initialize the dictionary to store final data
    record_dict = inizialize_record_dict(model)
    
    # Find all .h5ad or .h5ad.gz files in the specified folder
    files = [f for f in os.listdir(data_fold_path) if f.endswith(".h5ad.gz") or f.endswith(".h5ad")]
    print(f"Found {len(files)} files to process.")
    
    # Progress bar for files
    progress_bar_files = tqdm(files, desc="Processing files", unit="file")
    
    for file in progress_bar_files:
        file_path = os.path.join(data_fold_path, file)
        gene_ids, seq_data, labels = load_h5ad_file(file_path)
        
        # Add file origin to record_dict if not already present
        if 'file_origin' not in record_dict:
            record_dict['file_origin'] = []
        
        # Tokenize cell data
        tokenized_data = tokenizer.tokenize_cell_vectors(seq_data, gene_ids)
        
        # --- START: Code to verify tokenizer output ---
        for i, (cell_tokens_sample, cell_values_sample) in enumerate(tokenized_data):
            if i >= 1: # Print details for only the first cell
                break

        # --- END: Code to verify tokenizer output ---

        # Progress bar for cells within each file
        progress_bar_cells = tqdm(enumerate(tokenized_data), total=len(tokenized_data), desc=f"Processing cells in {file}", leave=False)
        
        # Process one cell at a time
        for cell_idx, (cell_tokens, cell_values) in progress_bar_cells:
            # Dictionary to store the input to the transformer for the current cell
            transformer_input_storage = {} 

            # Pre-forward hook function to capture the input to the transformer module
            def get_pre_forward_hook():
                def pre_hook(module, input):
                    # input is a tuple of (args, kwargs)
                    # The first argument (input[0]) should be the hidden_states tensor
                    # This tensor should have the shape (batch_size, sequence_length, embedding_dimension)
                    transformer_input_tensor = input[0]
                    
                    # Store the input tensor. Detach it to avoid memory issues if not used later.
                    transformer_input_storage['input_embedding'] = transformer_input_tensor.detach().cpu()
                    return input # Must return input
                return pre_hook

            # Register the pre-forward hook for the entire transformer module
            # This hook will be called BEFORE model.transformer's forward method
            hook = model.transformer.register_forward_pre_hook(get_pre_forward_hook())
                
            # Move tensors to device (GPU/CPU)
            cell_tokens = cell_tokens.to(device)
            cell_values = cell_values.to(device)
            
            # Create attention mask (assuming 0 is the padding value)
            attention_mask = (cell_values != 0).to(device)
            
            with torch.no_grad():
                # Perform the model's forward pass. The pre-hook will populate transformer_input_storage.
                _ = model(cell_tokens.unsqueeze(0), cell_values.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0))
            
           # After the forward pass, process and store the transformer input embedding
            # Add label and file origin for the current cell
            record_dict['label'].append(labels[cell_idx].tolist()) # Assuming labels is indexable by cell_idx
            record_dict['file_origin'].append(file)

            # Extract and store the transformer input embedding
            if 'input_embedding' in transformer_input_storage:
                input_embedding = transformer_input_storage['input_embedding']
                
                # Convert to NumPy array and remove the batch dimension (if present)
                processed_input_embedding = input_embedding.squeeze(0).numpy()
                
                # --- MODIFICA QUI PER IGNORARE IL TOKEN [CLS] ---
                # Se la lunghezza della sequenza è 5001, prendiamo dal secondo elemento (indice 1) in poi
                if processed_input_embedding.shape[0] == 5001:
                    processed_input_embedding = processed_input_embedding[1:] # Ignora il primo embedding (CLS)
                # --------------------------------------------------
                
                record_dict['transformer_input_embedding'].append(processed_input_embedding.tolist())
            else:
                print(f"Warning: No transformer input embedding captured for cell {cell_idx} in file {file}. Appending empty list.")
                record_dict['transformer_input_embedding'].append([]) # Append an empty list as a placeholder

            # Remove the hook immediately after processing the current cell's outputs
            hook.remove()
            
    # Create a Dataset object from the final dictionary
    dataset = Dataset.from_dict(record_dict)
    return dataset
