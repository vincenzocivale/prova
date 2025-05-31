import os
import pandas as pd
import numpy as np
from datasets import Dataset
from pathlib import Path
import torch
import scanpy as sc

def load_h5ad_file(file_path):

    adata = sc.read_h5ad(file_path) 

    gene_ids = adata.var_names.to_numpy()
    seq_data = adata.X
    labels =  adata.obs['predicted_label']

    return gene_ids, seq_data, labels

def inizialize_record_dict(model):

    layer_embeddings = {'label': []}

    for i in range(len(model.transformer.layers)):
        layer_embeddings[f'layer_{i}'] = []

    return layer_embeddings

def create_dataset(data_fold_path, tokenizer, model):

    record_dict = inizialize_record_dict(model)
    print(f"Found {len(os.listdir(data_fold_path))} files")
    from tqdm import tqdm
    files = [f for f in os.listdir(data_fold_path) if f.endswith(".h5ad.gz") or f.endswith(".h5ad")]
    progress_bar = tqdm(files, desc="Processing files", unit="file")
    for file in os.listdir(data_fold_path):
        if file.endswith(".h5ad.gz") or file.endswith(".h5ad"):

            gene_ids, seq_data, labels = load_h5ad_file(os.path.join(data_fold_path, file))

            record_dict['label'].append(labels)

            
            tokenized_data = tokenizer.tokenize_cell_vectors(seq_data, gene_ids)

            
            for cell_tokens, cell_values in tokenized_data:
                attention_mask = torch.tensor([v != 0 for v in cell_values], dtype=torch.bool)

                intermediate_outputs = {} # reset output intermedi

                with torch.no_grad():
                    _ = model(cell_tokens, cell_values, attention_mask=attention_mask)

                for layer, embedding in intermediate_outputs.items():
                    layer_num = int(layer.split('_')[1])
                    record_dict[f"layer_{layer_num}"].append(embedding)
        
    dataset = Dataset.from_dict(record_dict)
    
    return dataset

                
            

