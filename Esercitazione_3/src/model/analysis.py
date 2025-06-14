from tqdm import tqdm
import torch
from datasets import Dataset
import numpy as np
from typing import Dict, List
import torch.nn.functional as F
import pandas as pd


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
    
    device = embeddings.device
    target_values = target_values.to(device)
    masked_positions = masked_positions.to(device)
    non_padded_positions = non_padded_positions.to(device)
    
    if embeddings.dim() == 3:
        if embeddings.size(-1) != 1:
            predictions = embeddings.mean(dim=-1)
        else:
            predictions = embeddings.view(embeddings.size(0), embeddings.size(1))
    elif embeddings.dim() == 2:
        predictions = embeddings
    else:
        raise ValueError(f"Unexpected embeddings dimensions: {embeddings.shape}")
    
    if skip_cell_embedding:
        predictions = predictions[:, 1:]
        target_values = target_values[:, 1:]
        masked_positions = masked_positions[:, 1:]
        non_padded_positions = non_padded_positions[:, 1:]
    
    min_seq_len = min(predictions.size(1), target_values.size(1), 
                      masked_positions.size(1), non_padded_positions.size(1))
    
    predictions = predictions[:, :min_seq_len]
    target_values = target_values[:, :min_seq_len]
    masked_positions = masked_positions[:, :min_seq_len]
    non_padded_positions = non_padded_positions[:, :min_seq_len]
    
    eval_mask = masked_positions & non_padded_positions
    
    if eval_mask.sum() == 0:
        return {"mse": float('inf'), "mae": float('inf'), "pearson": 0.0}
    
    if predictions.dim() == 1:
        predictions = predictions.unsqueeze(0)
    elif predictions.dim() > 2:
        predictions = predictions.view(predictions.size(0), -1)
    
    if predictions.dim() == 1:
        predictions = predictions.unsqueeze(0)
    if target_values.dim() == 1:
        target_values = target_values.unsqueeze(0)

    pred_values = predictions[eval_mask]
    true_values = target_values[eval_mask]
    
    mse = F.mse_loss(pred_values, true_values).item()
    mae = F.l1_loss(pred_values, true_values).item()
    
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
    Evaluate all layer embeddings on the dataset and return average metrics per layer.
    
    Args:
        dataset: Hugging Face DatasetDict with 'train' field
        skip_cell_embedding: Whether to skip the first token
        include_zero_genes: Whether to include genes with expression 0
        pad_value: Padding value to ignore

    Returns:
        DataFrame with mean and std metric values per layer
    """
    
    layer_keys = [key for key in dataset.features if key.startswith("transformer_layer_")]
    
    input_values = torch.stack([torch.tensor(x, dtype=torch.float32) for x in dataset['label']])
    layer_embeddings = {
        layer: torch.stack([torch.tensor(x, dtype=torch.float32) for x in dataset[layer]])
        for layer in layer_keys
    }

    n_cells = input_values.shape[0]
    all_results = []

    for cell_idx in tqdm(range(n_cells), desc="Evaluating cells"):
        
        cell_input = input_values[cell_idx:cell_idx+1]
        
        seq_len_after_skip = cell_input.shape[1] - 1 if skip_cell_embedding else cell_input.shape[1]
        if seq_len_after_skip <= 0:
            continue

        if include_zero_genes:
            masked_values = torch.full_like(cell_input, True, dtype=torch.bool)
        else:
            masked_values = cell_input > 0

        non_padded_values = cell_input != pad_value

        for layer_name, layer_tensor in layer_embeddings.items():
            cell_embeddings = layer_tensor[cell_idx:cell_idx+1]

            emb_seq_len_after_skip = cell_embeddings.shape[1] - 1 if skip_cell_embedding else cell_embeddings.shape[1]
            if emb_seq_len_after_skip <= 0:
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

        if skip_cell_embedding:
            baseline_input = cell_input[:, 1:]
            baseline_masked = masked_values[:, 1:]
        else:
            baseline_input = cell_input
            baseline_masked = masked_values
            
        mean_value = baseline_input[baseline_masked].mean().item() if baseline_masked.sum() > 0 else 0
        mean_predictions = torch.full_like(baseline_input, mean_value)

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

    df = pd.DataFrame(all_results)
    mean_results = df.groupby('layer')[['mse', 'mae', 'pearson']].agg(['mean', 'std']).round(4)
    mean_results.columns = ['_'.join(col) for col in mean_results.columns]

    return mean_results.reset_index()


def combine_embeddings(embeddings_dict: Dict[str, torch.Tensor], 
                      combination_method: str = "mean",
                      layer_weights: Dict[str, float] = None) -> torch.Tensor:
    """
    Combine embeddings from different layers using various strategies
    
    Args:
        embeddings_dict: Dict with layer_name -> embedding tensor
        combination_method: "mean", "weighted_mean", "concat", "attention", "last_n"
        layer_weights: Weights for weighted_mean (optional)
    
    Returns:
        Combined embeddings tensor
    """
    
    embeddings_list = list(embeddings_dict.values())
    layer_names = list(embeddings_dict.keys())
    
    if combination_method == "mean":
        combined = torch.stack(embeddings_list, dim=0).mean(dim=0)
        
    elif combination_method == "weighted_mean":
        if layer_weights is None:
            layer_weights = {name: (i+1)/len(layer_names) 
                           for i, name in enumerate(layer_names)}
        
        weighted_embeddings = []
        for name, emb in embeddings_dict.items():
            weight = layer_weights.get(name, 1.0)
            weighted_embeddings.append(emb * weight)
        
        combined = torch.stack(weighted_embeddings, dim=0).sum(dim=0)
        total_weight = sum(layer_weights.get(name, 1.0) for name in layer_names)
        combined = combined / total_weight
        
    elif combination_method == "concat":
        if embeddings_list[0].dim() == 3:
            combined = torch.cat(embeddings_list, dim=-1)
        else:
            expanded = [emb.unsqueeze(-1) for emb in embeddings_list]
            combined = torch.cat(expanded, dim=-1)
            
    elif combination_method == "last_n":
        n_layers = min(4, len(embeddings_list))
        last_embeddings = embeddings_list[-n_layers:]
        combined = torch.stack(last_embeddings, dim=0).mean(dim=0)
        
    elif combination_method == "attention":
        if embeddings_list[0].dim() == 2:
            embeddings_list = [emb.unsqueeze(-1) for emb in embeddings_list]
            
        stacked = torch.stack(embeddings_list, dim=0)
        norms = torch.norm(stacked, dim=-1, keepdim=True)
        attention_weights = F.softmax(norms, dim=0)
        combined = (stacked * attention_weights).sum(dim=0)
        
    else:
        raise ValueError(f"Unknown combination method: {combination_method}")
    
    return combined




from typing import List, Dict
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

def evaluate_combined_embeddings(
    dataset,
    combination_methods: List[str] = ["mean", "weighted_mean", "last_n", "attention"],
    layer_groups: Dict[str, List[str]] = None,
    skip_cell_embedding: bool = True,
    include_zero_genes: bool = False,
    pad_value: int = -1
) -> pd.DataFrame:
    """
    Evaluate combined embeddings from multiple layers.

    Args:
        dataset: Hugging Face DatasetDict
        combination_methods: List of combination methods to test
        layer_groups: Dict of group_name -> list of layers to combine
        skip_cell_embedding: Whether to skip the first token
        include_zero_genes: Whether to include zero-expression genes
        pad_value: Padding value to ignore

    Returns:
        DataFrame with results for each combination
    """

    layer_keys = [key for key in dataset.features if key.startswith("transformer_layer_")]

    if layer_groups is None:
        n_layers = len(layer_keys)
        layer_groups = {
            "all_layers": layer_keys,
            "first_half": layer_keys[:n_layers // 2],
            "second_half": layer_keys[n_layers // 2:],
            "last_4": layer_keys[-4:] if n_layers >= 4 else layer_keys,
            "first_last": [layer_keys[0], layer_keys[-1]],
            "middle_layers": layer_keys[n_layers // 4 : 3 * n_layers // 4] if n_layers >= 4 else layer_keys[1:-1]
        }

    input_values = torch.stack([torch.tensor(x, dtype=torch.float32) for x in dataset['label']])
    layer_embeddings = {
        layer: torch.stack([torch.tensor(x, dtype=torch.float32) for x in dataset[layer]])
        for layer in layer_keys
    }

    n_cells = input_values.shape[0]
    all_results = []

    for cell_idx in tqdm(range(n_cells), desc="Evaluating combined embeddings"):
        cell_input = input_values[cell_idx : cell_idx + 1]

        seq_len_after_skip = cell_input.shape[1] - 1 if skip_cell_embedding else cell_input.shape[1]
        if seq_len_after_skip <= 0:
            continue

        if include_zero_genes:
            masked_values = torch.full_like(cell_input, True, dtype=torch.bool)
        else:
            masked_values = cell_input > 0

        non_padded_values = cell_input != pad_value

        cell_embeddings = {layer: layer_embeddings[layer][cell_idx : cell_idx + 1] for layer in layer_keys}

        for group_name, layers_to_combine in layer_groups.items():
            valid_layers = [l for l in layers_to_combine if l in cell_embeddings]
            if not valid_layers:
                continue

            embeddings_subset = {l: cell_embeddings[l] for l in valid_layers}

            for method in combination_methods:
                try:
                    combined_embeddings = combine_embeddings(embeddings_subset, combination_method=method)

                    metrics = calculate_gene_expr_metrics(
                        embeddings=combined_embeddings,
                        target_values=cell_input,
                        masked_positions=masked_values,
                        non_padded_positions=non_padded_values,
                        skip_cell_embedding=skip_cell_embedding,
                    )

                    metrics.update(
                        {
                            "combination_method": method,
                            "layer_group": group_name,
                            "n_layers_combined": len(valid_layers),
                            "layers_used": ",".join(valid_layers),
                            "cell_idx": cell_idx,
                        }
                    )

                    all_results.append(metrics)

                except Exception as e:
                    print(f"Error with {group_name}/{method}: {e}")
                    continue

    df = pd.DataFrame(all_results)

    if len(df) == 0:
        return pd.DataFrame()

    summary = df.groupby(["combination_method", "layer_group"])[["mse", "mae", "pearson"]].agg(["mean", "std"]).round(4)
    summary.columns = ["_".join(col) for col in summary.columns]

    return summary.reset_index()



def create_custom_layer_weights(layer_names: List[str], weight_strategy: str = "linear") -> Dict[str, float]:
    """
    Create custom weights for layers.

    Args:
        layer_names: List of layer names
        weight_strategy: "linear", "exponential", "gaussian", or "custom"

    Returns:
        Dict with weights per layer
    """

    n_layers = len(layer_names)

    if weight_strategy == "linear":
        weights = [(i + 1) / n_layers for i in range(n_layers)]

    elif weight_strategy == "exponential":
        weights = [np.exp(i / n_layers) for i in range(n_layers)]
        weights = [w / sum(weights) for w in weights]

    elif weight_strategy == "gaussian":
        center = n_layers * 0.75
        weights = [
            np.exp(-0.5 * ((i - center) / (n_layers * 0.2)) ** 2) for i in range(n_layers)
        ]
        weights = [w / sum(weights) for w in weights]

    else:
        weights = [1.0 / n_layers] * n_layers

    return dict(zip(layer_names, weights))


def run_comprehensive_evaluation(dataset):
    """
    Run full evaluation with multiple combinations.
    """

    print("1. Single layer evaluation...")
    single_results = evaluate_dataset_by_layer(dataset)

    print("\n2. Multiple layer combinations evaluation...")
    combined_results = evaluate_combined_embeddings(dataset)

    return {
        "single_layer_results": single_results,
        "combined_results": combined_results,
    }
