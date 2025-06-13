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
    
    return gene_ids, seq_data[:5000], labels[:5000] # Limito a 100 cellule per file per velocità

def create_dataset(data_fold_path, tokenizer, model):
    """
    Creates a dataset by extracting gene expression predictions from the intermediate embeddings
    of each transformer layer of a model, along with the tokenized input values.
    
    Args:
        data_fold_path (str): Path to the folder containing .h5ad files.
        tokenizer: The tokenizer object to convert cell vectors into tokens.
        model (torch.nn.Module): The PyTorch model from which to extract embeddings.
        
    Returns:
        datasets.Dataset: A Dataset object containing labels, file origin,
                          tokenized input values, and the gene expression predictions
                          from each transformer layer.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    record_dict = {
        'label': [],
        'file_origin': [],
        'tokenized_values': [] # Nuova colonna per i valori tokenizzati
    }
    
    num_layers = len(model.transformer.layers)
    # Prepara le colonne per gli output di ciascun layer Transformer
    for i in range(num_layers):
        record_dict[f'transformer_layer_{i}_output'] = []
    record_dict['transformer_full_output'] = [] # L'output dell'ultimo layer, rinominato

    # Cattura il decoder di espressione del modello per riutilizzarlo come "projection head"
    expr_decoder_head = model.expr_decoder.fc

    files = [f for f in os.listdir(data_fold_path) if f.endswith(".h5ad.gz") or f.endswith(".h5ad")]
    print(f"Found {len(files)} files to process.")
    
    progress_bar_files = tqdm(files, desc="Processing files", unit="file")
    
    for file in progress_bar_files:
        file_path = os.path.join(data_fold_path, file)
        gene_ids, seq_data, labels = load_h5ad_file(file_path)
        print(f"Processing file: {file} with {seq_data.shape[0]} cells and {seq_data.shape[1]} genes.")
        
        tokenized_data = tokenizer.tokenize_cell_vectors(seq_data, gene_ids)
        
        progress_bar_cells = tqdm(enumerate(tokenized_data), total=len(tokenized_data), 
                                  desc=f"Processing cells in {file}", leave=True)
        
        for cell_idx, (cell_tokens, cell_values) in progress_bar_cells:
            
            # Converte i tensori per il dispositivo corretto
            cell_tokens = cell_tokens.to(device)
            cell_values = cell_values.to(device)
            attention_mask = (cell_values != 0).to(device) # Crea la maschera di attenzione

            with torch.no_grad():
                # Memorizza i valori di espressione genica tokenizzati originali
                # Questi sono i valori che il modello ha effettivamente "visto" e su cui basa le predizioni.
                # Sono la tua base per il confronto con gli output di ciascun layer.
                record_dict['tokenized_values'].append(cell_values.cpu().numpy().tolist())

                # FASE 1: Calcola gli embedding di input al Transformer
                # Questi passaggi replicano la parte iniziale del forward pass del modello
                gene_embeddings = model.gene_encoder.embedding(cell_tokens.unsqueeze(0))
                gene_embeddings = model.gene_encoder.enc_norm(gene_embeddings)

                value_embeddings = model.value_encoder.linear1(cell_values.unsqueeze(0).unsqueeze(-1))
                value_embeddings = model.value_encoder.linear2(value_embeddings)
                value_embeddings = model.value_encoder.norm(value_embeddings)
                value_embeddings = model.value_encoder.dropout(value_embeddings)

                # Combina gli embedding di gene e valore per l'input al Transformer
                current_embeddings = gene_embeddings + value_embeddings
                
                # FASE 2: Itera attraverso i layer del Transformer
                # Per ogni strato, cattura l'output e lo proietta a valori di espressione genica.
                # L'output di ogni strato del Transformer mantiene la dimensione (batch_size, seq_len, 512).
                # Questo permette di analizzare come la rappresentazione contestuale evolve e quanto
                # è "pronta" per essere decodificata in espressione genica.
                for i, transformer_layer in enumerate(model.transformer.layers):
                    current_embeddings = transformer_layer(current_embeddings)
                    
                    # Applica la testina di proiezione (expr_decoder_head) per ottenere le predizioni di espressione genica.
                    # Questo trasforma i vettori di 512 dimensioni in valori scalari (1 dimensione) per ogni token,
                    # permettendo di confrontare direttamente queste "predizioni intermedie" con i valori tokenizzati di input.
                    predictions_from_layer = expr_decoder_head(current_embeddings).squeeze(-1)
                    
                    # Memorizza le predizioni dopo aver rimosso la dimensione del batch e convertito in lista
                    record_dict[f'transformer_layer_{i}_output'].append(predictions_from_layer.squeeze(0).cpu().numpy().tolist())
                
                # FASE 3: Estrai anche l'output finale del modello per confronto
                # current_embeddings a questo punto è l'output dell'ultimo strato del Transformer.
                final_predictions = expr_decoder_head(current_embeddings).squeeze(-1)
                record_dict['transformer_full_output'].append(final_predictions.squeeze(0).cpu().numpy().tolist())

            # Memorizza l'etichetta originale e l'origine del file per la cella corrente
            record_dict['label'].append(labels[cell_idx].tolist())
            record_dict['file_origin'].append(file)
    
    # Crea un oggetto Dataset finale
    dataset = Dataset.from_dict(record_dict)
    return dataset