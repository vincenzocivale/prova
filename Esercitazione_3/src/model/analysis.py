from tqdm import tqdm
import torch
from datasets import Dataset
import numpy as np
from typing import Dict
import torch.nn.functional as F
import pandas as pd

def predict_expr_per_layer(dataset, model, device='cuda'):
    results = []

    model.to(device)
    model.eval()

    progress_bar = tqdm(dataset, desc="Processing examples", unit="example")
    for example in progress_bar:
        row_results = {}
        for i in tqdm(range(12), desc="Processing layers", leave=False):  # 12 transformer layers
            embedding = example[f"transformer_layer_{i}"]  # [seq_len, hidden_dim]
            embedding = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(device)  # [1, seq_len, hidden_dim]

            with torch.no_grad():
                output = model.expr_decoder(embedding)["pred"]  # [1, seq_len, 1]

            print(f"embedding shape: {embedding.shape}")

            pred = output.squeeze(0).squeeze(-1).cpu().numpy()  # [seq_len]
            row_results[f"layer_{i}_expr_pred"] = pred

        if 'label' in example:
            row_results["label"] = example["label"]
        if 'file_origin' in example:
            row_results["file_origin"] = example["file_origin"]

        results.append(row_results)

    return Dataset.from_list(results)


def calculate_gene_expr_metrics(
    embeddings: torch.Tensor,
    target_values: torch.Tensor,
    masked_positions: torch.Tensor,
    non_padded_positions: torch.Tensor,
    skip_cell_embedding: bool = True
) -> Dict[str, float]:
    """
    Calculate gene expression prediction metrics
    
    Args:
        embeddings: Model embeddings [batch_size, seq_len, hidden_dim]
        target_values: True gene expression values [batch_size, seq_len]
        masked_positions: Boolean mask for positions to evaluate [batch_size, seq_len]
        non_padded_positions: Boolean mask for non-padded positions [batch_size, seq_len]
        skip_cell_embedding: Whether to skip cell embedding (first token)
    
    Returns:
        Dictionary with MSE, MAE, Pearson correlation metrics
    """
    
    # Convert to same device
    device = embeddings.device
    target_values = target_values.to(device)
    masked_positions = masked_positions.to(device)
    non_padded_positions = non_padded_positions.to(device)
    
    # Skip cell embedding if requested
    if skip_cell_embedding:
        embeddings = embeddings[:, 1:]
        target_values = target_values[:, 1:]
        masked_positions = masked_positions[:, 1:]
        non_padded_positions = non_padded_positions[:, 1:]
    
    # Create final mask (masked AND non-padded)
    eval_mask = masked_positions & non_padded_positions
    
    if eval_mask.sum() == 0:
        return {"mse": float('inf'), "mae": float('inf'), "pearson": 0.0}
    
    # Get predictions from embeddings (assuming linear projection needed)
    # If embeddings are already predictions, skip this step
    if embeddings.size(-1) != 1:
        # Simple linear projection from embedding dim to 1
        predictions = embeddings.mean(dim=-1, keepdim=True).squeeze(-1)
    else:
        predictions = embeddings.squeeze(-1)
    
   # Ensure predictions and target_values are same shape as eval_mask
    if predictions.dim() == 1:
        predictions = predictions.unsqueeze(0)
    if target_values.dim() == 1:
        target_values = target_values.unsqueeze(0)

    pred_values = predictions[eval_mask]
    true_values = target_values[eval_mask]

    
    # Calculate metrics
    mse = F.mse_loss(pred_values, true_values).item()
    mae = F.l1_loss(pred_values, true_values).item()
    
    # Pearson correlation
    if len(pred_values) > 1:
        pred_np = pred_values.detach().cpu().numpy()
        true_np = true_values.detach().cpu().numpy()
        pearson = np.corrcoef(pred_np, true_np)[0, 1]
        if np.isnan(pearson):
            pearson = 0.0
    else:
        pearson = 0.0
    
    return {
        "mse": mse,
        "mae": mae, 
        "pearson": pearson
    }


def evaluate_dataset_by_layer(
    dataset,
    skip_cell_embedding: bool = True,
    include_zero_genes: bool = False,
    pad_value: int = -1
) -> pd.DataFrame:
    """
    Valuta tutti gli embeddings di ciascun layer sul dataset Hugging Face
    e restituisce le metriche medie per layer (MSE, MAE, Pearson).
    
    Args:
        dataset: Hugging Face DatasetDict con campo 'train'
        skip_cell_embedding: Se saltare il primo token
        include_zero_genes: Se includere geni con espressione 0
        pad_value: Valore di padding da ignorare

    Returns:
        DataFrame con valori medi e std delle metriche per layer
    """
    
    # Ottieni nomi dei layer
    layer_keys = [key for key in dataset.features if key.startswith("transformer_layer_")]
    
    # Pre-estrai tutti i tensori dal dataset in RAM
    input_values = torch.stack([torch.tensor(x, dtype=torch.float32) for x in dataset['label']])
    layer_embeddings = {
        layer: torch.stack([torch.tensor(x, dtype=torch.float32) for x in dataset[layer]])
        for layer in layer_keys
    }

    n_cells = input_values.shape[0]
    all_results = []

    for cell_idx in tqdm(range(n_cells), desc="Evaluating cells"):
        cell_input = input_values[cell_idx:cell_idx+1]  # Keep batch dim
        
        # Skip se la sequenza diventa vuota dopo aver rimosso il token cell embedding
        if skip_cell_embedding and cell_input.shape[1] <= 1:
            continue

        if include_zero_genes:
            masked_values = torch.full_like(cell_input, True, dtype=torch.bool)
        else:
            masked_values = cell_input > 0

        non_padded_values = cell_input != pad_value

        for layer_name, layer_tensor in layer_embeddings.items():
            cell_embeddings = layer_tensor[cell_idx:cell_idx+1]

            if skip_cell_embedding and cell_embeddings.shape[1] <= 1:
                continue

            metrics = calculate_gene_expr_metrics(
                embeddings=cell_embeddings,
                target_values=cell_input,
                masked_positions=masked_values,
                non_padded_positions=non_padded_values,
                skip_cell_embedding=skip_cell_embedding
            )
            metrics['layer'] = layer_name
            metrics['cell_idx'] = cell_idx
            all_results.append(metrics)

        # Baseline (media per cella)
        mean_value = cell_input[masked_values].mean().item() if masked_values.sum() > 0 else 0
        mean_predictions = torch.full_like(cell_input, mean_value)

        baseline_metrics = calculate_gene_expr_metrics(
            embeddings=mean_predictions.unsqueeze(-1),
            target_values=cell_input,
            masked_positions=masked_values,
            non_padded_positions=non_padded_values,
            skip_cell_embedding=skip_cell_embedding
        )
        baseline_metrics['layer'] = 'mean_baseline'
        baseline_metrics['cell_idx'] = cell_idx
        all_results.append(baseline_metrics)

    # DataFrame completo
    df = pd.DataFrame(all_results)

    # Raggruppamento per layer
    mean_results = df.groupby('layer')[['mse', 'mae', 'pearson']].agg(['mean', 'std']).round(4)
    mean_results.columns = ['_'.join(col) for col in mean_results.columns]

    return mean_results.reset_index()
