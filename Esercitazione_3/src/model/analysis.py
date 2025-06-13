import tdqm 
import torch
from datasets import Dataset

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

        # facoltativo: aggiungi label o file_origin se vuoi mantenerli nel nuovo dataset
        if 'label' in example:
            row_results["label"] = example["label"]
        if 'file_origin' in example:
            row_results["file_origin"] = example["file_origin"]

        results.append(row_results)

    return Dataset.from_list(results)
