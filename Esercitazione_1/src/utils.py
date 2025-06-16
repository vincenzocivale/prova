from sklearn.metrics import accuracy_score
import numpy as np
from typing import Dict
import torch
from tqdm import tqdm
from src.models import distillation_loss

def compute_accuracy_metrics(predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    return {"accuracy": accuracy_score(labels, predictions)}

def train_student_with_distillation(student, dataloader, optimizer, device, temperature=4.0, alpha=0.7):
    student.train()
    total_loss = 0.0
    for x, y, teacher_logits in tqdm(dataloader, desc="Distill Training"):
        x, y, teacher_logits = x.to(device), y.to(device), teacher_logits.to(device)
        optimizer.zero_grad()
        student_logits = student(x)
        loss = distillation_loss(student_logits, teacher_logits, y, temperature, alpha)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)
