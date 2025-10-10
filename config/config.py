from abc import ABC, abstractmethod
from typing import Dict, Any, Protocol, Type, Optional, Tuple
import os
from dataclasses import dataclass
from enum import Enum

# =================== CONFIGURATION OBJECTS ===================

@dataclass
class DatasetConfig:
    """Configuration per dataset creation"""
    csv_path: str
    tokenizer_name: Optional[str] = None
    max_samples: Optional[int] = None
    imagenet_mapping_path: Optional[str] = None
    batch_size: int = 16

@dataclass
class ModelConfig:
    """Configuration per model loading"""
    model_path: str
    device: Optional[str] = None
    load_in_8bit: bool = False

@dataclass
class TaskConfig:
    """Configuration per task setup"""
    num_classes: int
    temperature: float = 3.0
    alpha: float = 0.8
    learning_rate: float = 1e-4
    epochs: int = 3
