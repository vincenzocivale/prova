# Deep Learning Applications - Laboratory 1: CNNs and ResNets
## Deep Residual Learning for Image Recognition

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Weights & Biases](https://img.shields.io/badge/Weights_&_Biases-FFBE00?logo=weightsandbiases&logoColor=white)](https://wandb.ai)

## 📋 Panoramica del Progetto

Questo laboratorio implementa e valuta diverse architetture di reti neurali per verificare le scoperte fondamentali del paper ResNet di He et al. (2016). Il progetto confronta reti neurali standard con architetture residuali su dataset MNIST e CIFAR-10, implementando anche tecniche di Knowledge Distillation per il trasferimento di conoscenza da modelli teacher a modelli student più piccoli.

### 🎯 Obiettivi Principali

1. **Replicare i risultati di ResNet**: Dimostrare che reti più profonde non garantiscono sempre prestazioni migliori
2. **Comparare architetture**: MLP standard vs ResidualMLP, CNN standard vs ResidualCNN
3. **Knowledge Distillation**: Trasferire conoscenza da modelli grandi a modelli compatti
4. **Training Pipeline**: Implementare un trainer modulare ispirato a HuggingFace Transformers

## 🏗️ Architettura del Progetto

```
Esercitazione_1/
├── Lab1_CNNs.ipynb          # Notebook principale con esperimenti
├── README.md                # Documentazione (questo file)  
├── requirements.txt         # Dipendenze Python
├── data/                    # Dataset MNIST e CIFAR-10
│   ├── MNIST/
│   └── cifar-10-batches-py/
├── models/                  # Modelli salvati
│   └── rescnn_ch16_d4_c10_teacher.pth
├── src/                     # Codice sorgente modulare
│   ├── __init__.py
│   ├── config.py           # Configurazioni di training
│   ├── data.py             # Gestione dataset e transforms
│   ├── models.py           # Definizione architetture
│   ├── trainer.py          # Training pipeline (ispirato a HuggingFace)
│   ├── utils.py            # Funzioni di utilità
│   └── experiments.py      # Configurazioni esperimenti
└── wandb/                  # Logs Weights & Biases
```

## 🧠 Modelli Implementati

### 1. Multi-Layer Perceptron (MLP)
- **Classe**: `MLP`
- **Caratteristiche**: Architettura fully-connected standard con dropout
- **Parametri configurabili**: hidden_sizes, activation_fn, dropout_rate

### 2. Residual MLP
- **Classe**: `ResidualMLP` 
- **Caratteristiche**: Blocchi residuali con skip connections
- **Innovazione**: Permette training di reti MLP molto profonde

### 3. Simple CNN
- **Classe**: `SimpleCNN`
- **Caratteristiche**: CNN standard con conv layers, batch normalization, pooling
- **Uso**: Baseline per comparazione con ResidualCNN

### 4. Residual CNN
- **Classe**: `ResidualCNN`
- **Caratteristiche**: Utilizza BasicBlock di torchvision.models.resnet
- **Vantaggi**: Skip connections per training di reti molto profonde

## 🚀 Training Pipeline - StreamlinedTrainer

Il `StreamlinedTrainer` è **ispirato alla classe Trainer di HuggingFace Transformers** e fornisce un'interfaccia unificata per il training di tutti i modelli.

### Caratteristiche Principali

#### ✨ Funzionalità Core
- **Configurazione Centralizzata**: Tutte le impostazioni in `TrainingConfig`
- **Early Stopping**: Prevenzione overfitting con monitoraggio metriche
- **Data Management**: Split automatico train/validation con stratificazione
- **Transforms Dinamiche**: Augmentation differenziata per train/test
- **Multi-GPU Support**: Supporto CUDA, MPS, CPU

#### 📊 Logging e Monitoraggio
- **Weights & Biases Integration**: Logging automatico metriche e modelli
- **Progress Tracking**: Barre di progresso con tqdm
- **Model Checkpointing**: Salvataggio automatico best model
- **Custom Metrics**: Support per metriche personalizzate

#### 🔧 Ottimizzatori Supportati
- Adam (default)
- SGD con momentum



## 🧪 Esperimenti Implementati

### Exercise 1.1-1.2: MLP vs ResidualMLP su MNIST
- **Dataset**: MNIST (28x28 grayscale)
- **Architetture**: MLP standard vs ResidualMLP
- **Profondità**: Shallow (2 layer), Medium (4 layer), Deep (8 layer)
- **Obiettivo**: Dimostrare vantaggi skip connections per reti profonde

### Exercise 1.3: CNN vs ResidualCNN su CIFAR-10
- **Dataset**: CIFAR-10 (32x32 RGB)
- **Architetture**: SimpleCNN vs ResidualCNN
- **Profondità**: 2, 3, 4 stage
- **Data Augmentation**: RandomRotation, RandomAffine, Normalization

### Exercise 2.2: Knowledge Distillation
- **Teacher Model**: ResidualCNN profonda (parametri: ~500K)
- **Student Model**: SimpleCNN compatta (parametri: ~50K)
- **Tecnica**: Distillation loss con temperature scaling
- **Obiettivo**: Trasferimento knowledge per modelli efficienti

## 📈 Metriche e Risultati

### Metriche Monitorate
- **Accuracy**: Precisione su test set
- **Loss**: Cross-entropy loss
- **Training Time**: Tempo per epoca
- **Model Size**: Numero parametri

### Knowledge Distillation Performance
- **Student Baseline**: ~XX% accuracy
- **Student + Distillation**: ~XX% accuracy  
- **Teacher Model**: ~XX% accuracy
- **Improvement**: +X.X% dalla distillation

## 🔬 Grafici e Dati da Recuperare da Weights & Biases

Per completare la documentazione, recupera questi grafici/dati da WandB:

### 📊 Grafici Essenziali

#### 1. **Comparison Charts - MLP Experiments**
- **Percorso WandB**: `MLP_vs_ResidualMLP_Comparison` project
- **Grafici necessari**:
  - Training/Validation Loss curves per depth
  - Test Accuracy comparison (bar chart)
  - Training time vs depth
  - Number of parameters vs performance

#### 2. **CNN Architecture Comparison**  
- **Percorso WandB**: `CNN_Comparison_CIFAR10` project
- **Grafici necessari**:
  - Validation accuracy curves (SimpleCNN vs ResidualCNN)
  - Loss curves per depth (2, 3, 4 stages)
  - Final test accuracy comparison
  - Training convergence speed

#### 3. **Knowledge Distillation Results**
- **Percorso WandB**: `dl-lab1-knowledge-distillation` project  
- **Grafici necessari**:
  - Student baseline vs distilled student accuracy
  - Temperature parameter effects
  - Loss curves (hard loss, soft loss, combined)
  - Model size vs performance trade-off

#### 4. **Training Dynamics**
- **Metriche generali**:
  - Learning rate schedules
  - Gradient norms durante training
  - Early stopping trigger points
  - Memory usage per model

### 📋 Tabelle Riassuntive da Creare

#### Performance Summary Table
| Model | Dataset | Params | Test Acc | Train Time |
|-------|---------|--------|----------|------------|
| MLP-shallow | MNIST | XXK | XX.X% | XXs |
| ResidualMLP-shallow | MNIST | XXK | XX.X% | XXs |
| SimpleCNN-d2 | CIFAR-10 | XXK | XX.X% | XXs |
| ResidualCNN-d2 | CIFAR-10 | XXK | XX.X% | XXs |

#### Knowledge Distillation Summary
| Model | Params | Accuracy | Improvement |
|-------|--------|----------|-------------|
| Teacher (ResidualCNN) | ~500K | XX.X% | - |
| Student Baseline | ~50K | XX.X% | - |
| Student + Distillation | ~50K | XX.X% | +X.X% |

