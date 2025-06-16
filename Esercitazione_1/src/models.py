import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce

class MLP(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_sizes: list[int], activation_fn: nn.Module = nn.ReLU, dropout_rate: float = 0.0):
        super().__init__()
        if not hidden_sizes:
            raise ValueError("hidden_sizes cannot be empty")
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.activation_fn = activation_fn
        self.dropout_rate = dropout_rate
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                activation_fn(),
                nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
            ])
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.network(x)
    def __str__(self) -> str:
        hidden_str = "_".join(map(str, self.hidden_sizes))
        activation_name = self.activation_fn.__name__.lower()
        name = f"mlp_in{self.input_size}_h{hidden_str}_out{self.output_size}_{activation_name}"
        if self.dropout_rate > 0:
            name += f"_drop{str(self.dropout_rate).replace('.', '')}"
        return name

class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_layers: int, activation_fn: nn.Module, dropout_rate: float):
        super().__init__()
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(in_dim, out_dim))
            else:
                layers.append(nn.Linear(out_dim, out_dim))
            layers.append(activation_fn())
            layers.append(nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity())
        self.layers = nn.Sequential(*layers)
        self.use_skip = in_dim == out_dim
        if not self.use_skip:
            self.skip_proj = nn.Linear(in_dim, out_dim)
        else:
            self.skip_proj = nn.Identity()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.skip_proj(x) + self.layers(x)

class ResidualMLP(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_sizes: list[int], layers_per_block: int = 2, activation_fn: nn.Module = nn.ReLU, dropout_rate: float = 0.0):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.layers_per_block = layers_per_block
        self.activation_fn = activation_fn
        self.dropout_rate = dropout_rate
        layers = []
        prev_dim = input_size
        for h in hidden_sizes:
            layers.append(ResidualBlock(prev_dim, h, layers_per_block, activation_fn, dropout_rate))
            prev_dim = h
        self.residual_blocks = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, output_size)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.residual_blocks(x)
        return self.output_layer(x)
    def __str__(self) -> str:
        hidden_str = "_".join(map(str, self.hidden_sizes))
        activation_name = self.activation_fn.__name__.lower()
        name = f"resmlp_in{self.input_size}_h{hidden_str}_out{self.output_size}_{activation_name}"
        if self.dropout_rate > 0:
            name += f"_drop{str(self.dropout_rate).replace('.', '')}"
        return name

class SimpleCNN(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 10, depth: int = 3, initial_channels: int = 16):
        super().__init__()
        self.depth = depth
        self.initial_channels = initial_channels
        self.in_channels = in_channels
        self.num_classes = num_classes
        layers = []
        current_channels = initial_channels
        layers.extend([
            nn.Conv2d(in_channels, current_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(current_channels),
            nn.ReLU(inplace=True)
        ])
        for i in range(depth):
            next_channels = current_channels * 2
            layers.extend([
                nn.Conv2d(current_channels, next_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(next_channels),
                nn.ReLU(inplace=True)
            ])
            current_channels = next_channels
        self.features = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(current_channels, num_classes)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
    def __str__(self) -> str:
        return f"simplecnn_ch{self.initial_channels}_d{self.depth}_c{self.num_classes}"

from torchvision.models.resnet import BasicBlock
class ResidualCNN(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 10, depth: int = 3, initial_channels: int = 16):
        super().__init__()
        self.depth = depth
        self.initial_channels = initial_channels
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, initial_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(initial_channels),
            nn.ReLU(inplace=True)
        )
        self.stages = nn.Sequential()
        current_channels = initial_channels
        for i in range(depth):
            out_channels = current_channels * 2
            stage = self._make_stage(current_channels, out_channels, stride=2)
            self.stages.add_module(f"stage_{i+1}", stage)
            current_channels = out_channels
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(current_channels, num_classes)
    def _make_stage(self, in_channels: int, out_channels: int, stride: int = 1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        return nn.Sequential(
            BasicBlock(in_channels, out_channels, stride=stride, downsample=downsample),
            BasicBlock(out_channels, out_channels)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
    def __str__(self) -> str:
        return f"rescnn_ch{self.initial_channels}_d{self.depth}_c{self.num_classes}"
    
def distillation_loss(student_logits, teacher_logits, labels, temperature=4.0, alpha=0.7):
    """
    Calcola la loss per la knowledge distillation.
    student_logits: output del modello studente (logits)
    teacher_logits: output del modello teacher (logits)
    labels: target reali
    temperature: temperatura per la softmax
    alpha: peso tra soft e hard loss
    """
    # Soft targets
    soft_teacher = F.log_softmax(teacher_logits / temperature, dim=1)
    soft_student = F.log_softmax(student_logits / temperature, dim=1)
    kl_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean', log_target=True) * (temperature ** 2)
    # Hard targets
    ce_loss = F.cross_entropy(student_logits, labels)
    return alpha * kl_loss + (1 - alpha) * ce_loss
