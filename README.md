## Knowledge Distillation Framework
A flexible PyTorch-based framework for knowledge distillation supporting multiple task types (image classification, text classification, tabular data) with automatic task detection and model compatibility handling.
# Description
This framework simplifies knowledge distillation by providing:

Multi-task support: Automatic detection and handling of image, text, and tabular classification tasks
Flexible architecture: Modular adapters for datasets, models, and task-specific distillation strategies
HuggingFace integration: Native support for transformer models with automatic tokenizer compatibility detection
Energy tracking: Built-in CodeCarbon integration for monitoring energy consumption

# Project Status
This project represents an exploration of generalized knowledge distillation across diverse architectures and task types. 
While the core framework successfully achieves functional automatic distillation across heterogeneous models proved more complex than initially anticipated due to architectural incompatibilities and the absence of standardized dataset formats. 
The codebase provides a solid foundation for task-specific distillation implementations and may serve as a reference for future work in this domain.

# Installation
bash# Clone repository
git clone <repository-url>
cd <repository-name>

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


# Install dependencies
pip install -r requirements.txt


# Usage
Basic Distillation Example
pythonfrom distiller import DistillerBridge

# Initialize distillation bridge
bridge = DistillerBridge(
    teacher_path="./models/teacher.pt",
    student_path="./models/student.pt",
    dataset_path="./datasets/train.csv",
    output_path="./output/",
    tokenizer_name="bert-base-uncased"  # For text tasks
)

# Run distillation
bridge.distill()
Text Classification (BERT)
pythonfrom transformers import AutoModelForSequenceClassification
from utils.save_model import save_model_to_pt

# Save HuggingFace model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
save_model_to_pt(model, "./models/bert.pt")

# Run self-distillation
bridge = DistillerBridge(
    teacher_path="./models/bert.pt",
    student_path="./models/bert.pt",
    dataset_path="./datasets/sst2.csv",
    output_path="./output/bert_distilled/",
    tokenizer_name="bert-base-uncased"
)
bridge.distill()
Image Classification (ResNet)
pythonimport torch
from torchvision.models import resnet18, resnet34

# Save models
torch.save(resnet34(num_classes=10), "./models/teacher.pt")
torch.save(resnet18(num_classes=10), "./models/student.pt")

# Run distillation
bridge = DistillerBridge(
    teacher_path="./models/teacher.pt",
    student_path="./models/student.pt",
    dataset_path="./datasets/cifar10.csv",
    output_path="./output/resnet_distilled/"
)
bridge.distill()
Configuration
Edit config/constants.py to adjust hyperparameters:
pythonTEMPERATURE = 2.0    # Distillation temperature
ALPHA = 0.9          # Weight for soft loss (1-ALPHA for hard loss)
BATCH_SIZE = 32      # Training batch size
LEARNING_RATE = 1e-4 # Optimizer learning rate
Dataset Format
CSV files with two columns:

Text classification: text,label
Image classification: image_path,label or base64_image,label
Tabular: feature1,feature2,...,label

Example:
csvtext,label
"This movie is great",1
"Terrible film",0
```

## Architecture
```
├── adapters/          # Dataset and model adapters
├── dataset_by_tasking/ # Task-specific handlers
├── factories/         # Component factories
├── config/            # Configuration files
├── utils/             # Utilities (logging, energy tracking)
├── distiller.py       # Main distillation engine
└── provider/          # Task providers and interfaces

# Key Features

Automatic task detection: Infers task type from dataset structure
Dynamic input filtering: Handles tokenizer compatibility (input_ids, attention_mask, token_type_ids)
Label mapping: Supports ImageNet mapping and automatic class indexing
Device management: Automatic GPU/CPU device handling
Energy tracking: CodeCarbon integration for sustainability metrics


# Authors
@andreaeliia @andrea_andrenucci



[Specify your license here]
Authors
[Your contact information]
