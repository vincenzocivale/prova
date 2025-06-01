import os
import pandas as pd
import numpy as np
from datasets import Dataset
from pathlib import Path
import torch
import scanpy as sc
from tqdm import tqdm

def load_h5ad_file(file_path):
    adata = sc.read_h5ad(file_path) 
    gene_ids = adata.var_names.to_numpy()
    seq_data = adata.X
    labels = adata.obs['predicted_label']
    return gene_ids, seq_data, labels

def inizialize_record_dict(model):
    layer_embeddings = {'label': []}
    for i in range(len(model.transformer.layers)):
        layer_embeddings[f'transformer_layer_{i}'] = []  # Cambiato il nome per matchare gli hook
    return layer_embeddings

def create_dataset(data_fold_path, tokenizer, model):
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()  # Set model to evaluation mode

    # Dizionario per memorizzare gli output intermedi
    intermediate_outputs = {}
    
    # Funzione hook per catturare gli output
    def get_hook(layer_idx):
        def hook(module, input, output):
            # Per i layer transformer, l'output potrebbe essere una tupla
            if isinstance(output, tuple):
                output = output[0]
            # Estraiamo i dati dal NestedTensor
            if hasattr(output, 'to_padded_tensor'):
                output = output.to_padded_tensor(0.0)  # Usiamo 0.0 come valore di padding
            intermediate_outputs[f'transformer_layer_{layer_idx}'] = output
        return hook
    
    # Registra gli hook per ogni layer
    hooks = []
    for idx, layer in enumerate(model.transformer.layers):
        hook = layer.register_forward_hook(get_hook(idx))
        hooks.append(hook)

    record_dict = inizialize_record_dict(model)
    
    files = [f for f in os.listdir(data_fold_path) if f.endswith(".h5ad.gz") or f.endswith(".h5ad")]
    print(f"Found {len(files)} files")
    
    progress_bar = tqdm(files, desc="Processing files", unit="file")
    for file in progress_bar:
        file_path = os.path.join(data_fold_path, file)
        gene_ids, seq_data, labels = load_h5ad_file(file_path)
        record_dict['label'].extend(labels)
        
        tokenized_data = tokenizer.tokenize_cell_vectors(seq_data, gene_ids)
        
        # Process one cell at a time
        for cell_tokens, cell_values in tqdm(tokenized_data, desc="Processing cells", leave=False):
            # Move tensors to device
            cell_tokens = cell_tokens.to(device)
            cell_values = cell_values.to(device)
            attention_mask = torch.tensor([v != 0 for v in cell_values], dtype=torch.bool).to(device)
            
            with torch.no_grad():
                # Forward pass
                _ = model(cell_tokens.unsqueeze(0), cell_values.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0))
                
                # Store embeddings for each layer
                for layer_idx in range(len(model.transformer.layers)):
                    layer_key = f'transformer_layer_{layer_idx}'
                    if layer_key in intermediate_outputs:
                        layer_output = intermediate_outputs[layer_key]
                        # Convertiamo il tensore in un array numpy in modo sicuro
                        if isinstance(layer_output, torch.Tensor):
                            layer_output = layer_output.detach().cpu().numpy()
                            record_dict[layer_key].append(layer_output[0])  # Prendiamo solo il primo elemento del batch

    # Rimuovi gli hook
    for hook in hooks:
        hook.remove()
    
    return record_dict

                
            

