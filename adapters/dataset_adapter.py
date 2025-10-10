import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import io
import base64
import os
import json
from config.constants import BATCH_SIZE
from transformers import AutoTokenizer
from typing import Dict, List, Set, Any


class BaseDatasetAdapter:
    """
    Adapter class for managing different types of datasets (image, text, tabular).
    Handles data loading, tokenization, label mapping, and dynamic input filtering.
    """
    
    def __init__(self, csv_path, tokenizer_name=None, max_samples=None, imagenet_mapping_path=None):
        """
        Initialize the dataset adapter.
        
        Args:
            csv_path: Path to the CSV file containing the dataset
            tokenizer_name: Name of the HuggingFace tokenizer to use
            max_samples: Maximum number of samples to load (for testing/debugging)
            imagenet_mapping_path: Path to ImageNet label mapping JSON file
        """
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.imagenet_mapping_path = imagenet_mapping_path
        
        # Track which inputs are supported by the model and tokenizer capabilities
        self.supported_inputs = None
        self.tokenizer_capabilities = None

        if max_samples:
            print(f"[INFO] Trimming dataset to max {max_samples} samples")
            self.df = self.df.head(max_samples)

        print(f"[INFO] Loaded dataset with shape: {self.df.shape}")
        self.mode = self.infer_mode()
        print(f"[INFO] Inferred mode: {self.mode}")
        
        # Initialize tokenizer if provided
        self.tokenizer = None
        if tokenizer_name:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            self.tokenizer_capabilities = self._detect_tokenizer_capabilities()

        # Setup label mapping (ImageNet or automatic)
        self.setup_label_mapping()

    def _detect_tokenizer_capabilities(self) -> Dict[str, bool]:
        """
        Dynamically detect which outputs the tokenizer can generate.
        Tests tokenizer to determine which keys it produces (input_ids, attention_mask, etc).
        
        Returns:
            Dictionary mapping output keys to boolean indicating if tokenizer supports them
        """
        if not self.tokenizer:
            return {}
            
        print(f"[DATASET] Detecting tokenizer capabilities...")
        
        test_texts = ["This is a test sentence."]
        capabilities = {}
        
        try:
            base_output = self.tokenizer(
                test_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=64
            )
            
            # Check which outputs are generated
            for key in ['input_ids', 'attention_mask', 'token_type_ids', 'position_ids']:
                capabilities[key] = key in base_output
                if capabilities[key]:
                    print(f"[DATASET] Tokenizer generates: {key}")
            
            return capabilities
            
        except Exception as e:
            print(f"[DATASET] Error detecting tokenizer: {e}")
            return {
                'input_ids': True,
                'attention_mask': True,
                'token_type_ids': False,
                'position_ids': False
            }
    
    def set_supported_inputs(self, supported_inputs: Set[str]):
        """
        Set which inputs are supported by the models.
        Called by TaskHandler after dynamic detection.
        
        Args:
            supported_inputs: Set of input keys that the model accepts
        """
        self.supported_inputs = supported_inputs
        print(f"[DATASET] Supported inputs set: {sorted(supported_inputs)}")

    def setup_label_mapping(self):
        """
        Configure label mapping - can be ImageNet-specific or automatically generated.
        """
        if self.imagenet_mapping_path and os.path.exists(self.imagenet_mapping_path):
            print(f"[INFO] Loading ImageNet mapping from: {self.imagenet_mapping_path}")
            self.load_imagenet_mapping()
        else:
            print(f"[INFO] Creating automatic mapping from dataset labels")
            self.create_automatic_mapping()
        
        self.apply_label_mapping()

    def get_dataset_info(self):
        """
        Return essential dataset information.
        
        Returns:
            Dictionary containing number of classes, class names, label mapping, task type, and sample count
        """
        return {
            'num_classes': len(self.label_to_idx),
            'class_names': list(self.label_to_idx.keys()),
            'label_mapping': self.label_to_idx,
            'task_type': self.mode,
            'num_samples': len(self.df)
        }

    def load_imagenet_mapping(self):
        """
        Load ImageNet mapping from JSON file.
        Expects file with 'label_to_idx' and 'idx_to_label' dictionaries.
        """
        with open(self.imagenet_mapping_path, 'r') as f:
            mapping_data = json.load(f)
        
        self.label_to_idx = mapping_data.get('label_to_idx', {})
        self.idx_to_label = mapping_data.get('idx_to_label', {})
        # Convert numeric keys of idx_to_label to int
        self.idx_to_label = {int(k): v for k, v in self.idx_to_label.items()}
        
        print(f"[INFO] ImageNet mapping loaded: {len(self.label_to_idx)} classes")

    def create_automatic_mapping(self):
        """
        Create automatic mapping from unique labels in the dataset.
        Assigns sequential indices to sorted unique labels.
        """
        unique_labels = self.df.iloc[:, 1].unique()
        self.label_to_idx = {}
        self.idx_to_label = {}
        
        for idx, label in enumerate(sorted(unique_labels)):
            self.label_to_idx[str(label)] = idx
            self.idx_to_label[idx] = str(label)
        
        print(f"[INFO] Automatic mapping created: {len(unique_labels)} classes")

    def apply_label_mapping(self):
        """
        Apply label mapping to the DataFrame.
        Converts string labels to numeric indices using the mapping.
        """
        # Convert labels to strings for mapping
        self.df.iloc[:, 1] = self.df.iloc[:, 1].astype(str)
        
        original_labels = self.df.iloc[:, 1].copy()
        mapped_labels = self.df.iloc[:, 1].map(self.label_to_idx)
        
        # Check for unmapped labels
        unmapped_mask = mapped_labels.isna()
        if unmapped_mask.any():
            unmapped_labels = original_labels[unmapped_mask].unique()
            print(f"[WARNING] Unmapped labels found: {unmapped_labels[:10]}...")
            # Assign 0 to unmapped labels
            mapped_labels = mapped_labels.fillna(0)
        
        self.df.iloc[:, 1] = mapped_labels.astype(int)
        print(f"[INFO] Labels mapped successfully. Range: {mapped_labels.min()}-{mapped_labels.max()}")

    def save_mapping(self, output_path):
        """
        Save current mapping to a JSON file.
        
        Args:
            output_path: Path where the mapping JSON will be saved
        """
        mapping_data = {
            'label_to_idx': self.label_to_idx,
            'idx_to_label': self.idx_to_label,
            'num_classes': len(self.label_to_idx)
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(mapping_data, f, indent=2)
        
        print(f"[INFO] Mapping saved to: {output_path}")

    def infer_mode(self):
        """
        Infer dataset mode (image, text, or tabular) based on first column content.
        
        Returns:
            String indicating mode: "image", "text", or "tabular"
        """
        first_col = self.df.iloc[:, 0].astype(str)
        if first_col.str.contains(r'\.(jpg|jpeg|png)$', case=False).any():
            return "image"
        elif first_col.str.startswith("data:image/").any():
            return "image"
        elif first_col.str.len().mean() > 30:
            return "text"
        else:
            return "tabular"

    def get_dataloader(self):
        """
        Get appropriate DataLoader based on inferred mode.
        
        Returns:
            PyTorch DataLoader for the detected dataset type
        """
        if self.mode == "image":
            return self.get_image_loader()
        elif self.mode == "text":
            return self.get_text_loader()
        else:
            return self.get_tabular_loader()

    def get_image_loader(self):
        """
        Create DataLoader for image datasets.
        Applies standard ImageNet transformations (resize, normalize).
        
        Returns:
            DataLoader with image preprocessing pipeline
        """
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        class ImageDataset(Dataset):
            def __init__(self, df, transform, label_to_idx):
                self.paths = df.iloc[:, 0].tolist()
                self.labels = df.iloc[:, 1].tolist()
                self.transform = transform
                self.label_to_idx = label_to_idx

            def __getitem__(self, idx):
                img_path = self.paths[idx]
                label = self.labels[idx]
                
                try:
                    if img_path.startswith("data:image"):
                        # Handle base64 encoded images
                        image = Image.open(io.BytesIO(base64.b64decode(img_path.split(",")[1]))).convert("RGB")
                    elif os.path.exists(img_path):
                        # Handle file paths
                        image = Image.open(img_path).convert("RGB")
                    else:
                        print(f"[WARNING] File not found: {img_path}")
                        image = Image.new('RGB', (224, 224), color='gray')
                except Exception as e:
                    print(f"[ERROR] Error loading image {img_path}: {e}")
                    image = Image.new('RGB', (224, 224), color='gray')
                
                if self.transform:
                    image = self.transform(image)
                
                return image, int(label)

            def __len__(self):
                return len(self.paths)

        dataset = ImageDataset(self.df, transform, self.label_to_idx)
        return DataLoader(
            dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=True,
            num_workers=4,
            pin_memory=torch.cuda.is_available()
        )

    def get_text_loader(self):
        """
        Create DataLoader for text datasets with dynamic input filtering.
        Tokenizes text and filters outputs based on model's supported inputs.
        
        Returns:
            DataLoader with custom collate function for text tokenization
        """
        if not self.tokenizer:
            raise ValueError("Tokenizer is required for text mode.")

        class TextDataset(Dataset):
            def __init__(self, df):
                self.texts = df.iloc[:, 0].astype(str).tolist()
                self.labels = df.iloc[:, 1].tolist()

            def __getitem__(self, idx):
                return self.texts[idx], self.labels[idx]

            def __len__(self):
                return len(self.texts)

        def dynamic_collate_fn(batch):
            texts, labels = zip(*batch)
            
            # Tokenize with all possible parameters
            tokenized = self.tokenizer(
                list(texts),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
                add_special_tokens=True
            )
            
            # Dynamic filtering based on supported inputs
            if self.supported_inputs is not None:
                filtered_tokenized = self._filter_tokenizer_output(tokenized)
            else:
                # Fallback: use conservative strategy
                filtered_tokenized = self._conservative_filter(tokenized)
            
            # Add labels
            filtered_tokenized['labels'] = torch.tensor(labels)
            
            return filtered_tokenized

        return DataLoader(
            TextDataset(self.df), 
            batch_size=BATCH_SIZE, 
            collate_fn=dynamic_collate_fn,
            shuffle=True
        )
    
    def _filter_tokenizer_output(self, tokenized_output: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Filter tokenizer output based on model's supported inputs.
        
        Args:
            tokenized_output: Raw tokenizer output dictionary
            
        Returns:
            Filtered dictionary containing only supported input keys
        """
        filtered_output = {}

        for key, tensor in tokenized_output.items():
            if key in self.supported_inputs:
                filtered_output[key] = tensor
        
        # Ensure essential inputs are present
        essential_inputs = ['input_ids', 'attention_mask']
        for essential in essential_inputs:
            if essential not in filtered_output and essential in tokenized_output:
                filtered_output[essential] = tokenized_output[essential]
        
        return filtered_output
    
    def _conservative_filter(self, tokenized_output: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Conservative filtering strategy when supported inputs are unknown.
        Uses only universally supported inputs.
        
        Args:
            tokenized_output: Raw tokenizer output dictionary
            
        Returns:
            Dictionary with safe, universally supported inputs
        """
        print(f"[DATASET] Using conservative filter...")
        
        # Inputs that work with most models
        safe_inputs = ['input_ids', 'attention_mask']
        
        filtered_output = {}
        for key in safe_inputs:
            if key in tokenized_output:
                filtered_output[key] = tokenized_output[key]
                print(f"[DATASET] Safe input included: {key}")
        
        # Check if we can include token_type_ids based on tokenizer capabilities
        if (self.tokenizer_capabilities and 
            self.tokenizer_capabilities.get('token_type_ids', False) and
            'token_type_ids' in tokenized_output):
            
            if self._test_token_type_ids_support():
                filtered_output['token_type_ids'] = tokenized_output['token_type_ids']
                print(f"[DATASET] token_type_ids included (tested)")
            else:
                print(f"[DATASET] token_type_ids excluded (test failed)")
        
        return filtered_output
    
    def _test_token_type_ids_support(self) -> bool:
        """
        Test if tokenizer actually generates valid token_type_ids.
        
        Returns:
            True if tokenizer properly supports token_type_ids, False otherwise
        """
        try:
            test_output = self.tokenizer(
                ["Test sentence one.", "Test sentence two."],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=32
            )
            
            if 'token_type_ids' in test_output:
                token_type_ids = test_output['token_type_ids']
                # Verify shape matches input_ids
                is_valid_shape = token_type_ids.shape == test_output['input_ids'].shape
                return is_valid_shape
            
            return False
            
        except Exception as e:
            print(f"[DATASET] token_type_ids test failed: {e}")
            return False

    def get_tabular_loader(self):
        """
        Create DataLoader for tabular datasets.
        Assumes all columns except last are features, last column is label.
        
        Returns:
            DataLoader for tabular data
        """
        class TabularDataset(Dataset):
            def __init__(self, df):
                self.x = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32)
                self.y = torch.tensor(df.iloc[:, -1].values, dtype=torch.long)

            def __getitem__(self, idx):
                return self.x[idx], self.y[idx]

            def __len__(self):
                return len(self.x)

        return DataLoader(TabularDataset(self.df), batch_size=BATCH_SIZE, shuffle=True)

    def get_generation_loader(self):
        """
        Create DataLoader for text generation tasks.
        Assumes first column is input text, second column is target text.
        
        Returns:
            DataLoader with input-target pairs for generation
        """
        if not self.tokenizer:
            raise ValueError("Tokenizer is required for generation mode.")

        class GenerationDataset(Dataset):
            def __init__(self, df):
                self.inputs = df.iloc[:, 0].astype(str).tolist()
                self.targets = df.iloc[:, 1].astype(str).tolist()
            
            def __getitem__(self, idx):
                return self.inputs[idx], self.targets[idx]
            
            def __len__(self):
                return len(self.inputs)

        def collate_fn(batch):
            inputs, targets = zip(*batch)
            
            # Tokenize inputs
            input_encoding = self.tokenizer(
                list(inputs),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            # Tokenize targets
            target_encoding = self.tokenizer(
                list(targets),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            return {
                'input_ids': input_encoding['input_ids'],
                'attention_mask': input_encoding['attention_mask'],
                'target_ids': target_encoding['input_ids'],
                'target_attention_mask': target_encoding['attention_mask']
            }

        dataset = GenerationDataset(self.df)
        return DataLoader(
            dataset, 
            batch_size=BATCH_SIZE, 
            collate_fn=collate_fn,
            shuffle=True
        )

    def get_num_classes(self):
        """
        Get the number of classes in the dataset.
        
        Returns:
            Integer number of unique classes
        """
        return len(self.label_to_idx)

    def print_mapping_info(self):
        """
        Print detailed information about class mapping.
        Shows number of classes, index range, and sample mappings.
        """
        print(f"[INFO] === MAPPING INFO ===")
        print(f"Numero classi: {len(self.label_to_idx)}")
        print(f"Range indici: 0 - {len(self.label_to_idx) - 1}")
        
        # Show some mapping examples
        sample_size = min(10, len(self.label_to_idx))
        sample_items = list(self.label_to_idx.items())[:sample_size]
        print(f"Esempi mapping (primi {sample_size}):")
        for label, idx in sample_items:
            print(f"  '{label}' -> {idx}")
        print(f"[INFO] ====================")

    def debug_tokenizer_compatibility(self, sample_texts: List[str] = None):
        """
        Debug tokenizer compatibility with various configurations.
        Tests different tokenizer settings to verify proper functionality.
        
        Args:
            sample_texts: Optional list of test texts. Uses default samples if None.
        """
        if not self.tokenizer:
            print("No tokenizer available")
            return
        
        if sample_texts is None:
            sample_texts = [
                "This is a short test.",
                "This is a longer test sentence with more words for testing purposes.",
                "Another example text."
            ]
        
        print(f"\nDEBUG TOKENIZER COMPATIBILITY")
        print(f"Tokenizer: {getattr(self.tokenizer, 'name_or_path', 'Unknown')}")
        print("-" * 50)
        
        try:
            # Test with different parameters
            test_configs = [
                {"padding": True, "truncation": True, "max_length": 64},
                {"padding": True, "truncation": True, "max_length": 128, "add_special_tokens": True},
                {"padding": "max_length", "truncation": True, "max_length": 32}
            ]
            
            for i, config in enumerate(test_configs):
                print(f"\nTest {i+1}: {config}")
                try:
                    output = self.tokenizer(
                        sample_texts,
                        return_tensors="pt",
                        **config
                    )
                    
                    print(f"  Output keys: {list(output.keys())}")
                    for key, tensor in output.items():
                        print(f"    {key}: {tensor.shape}")
                        
                except Exception as e:
                    print(f"  Config failed: {e}")
        
        except Exception as e:
            print(f"Debug failed: {e}")
    
    def get_tokenizer_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive tokenizer report.
        
        Returns:
            Dictionary containing tokenizer name, capabilities, supported inputs, vocab size, and max length
        """
        if not self.tokenizer:
            return {"error": "No tokenizer available"}
        
        report = {
            "tokenizer_name": getattr(self.tokenizer, 'name_or_path', 'Unknown'),
            "capabilities": self.tokenizer_capabilities,
            "supported_inputs": list(self.supported_inputs) if self.supported_inputs else None,
            "vocab_size": getattr(self.tokenizer, 'vocab_size', 'Unknown'),
            "max_length": getattr(self.tokenizer, 'model_max_length', 'Unknown')
        }
        
        return report


def create_imagenet_mapping_from_train_dir(train_dir, output_path):
    """
    Create ImageNet mapping from training directory structure.
    Each subdirectory name becomes a class label.
    
    Args:
        train_dir: Path to training directory containing class subdirectories
        output_path: Path where mapping JSON will be saved
        
    Returns:
        Dictionary containing the created mapping
    """
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    
    class_dirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    class_dirs.sort()
    
    label_to_idx = {}
    idx_to_label = {}
    
    for idx, class_dir in enumerate(class_dirs):
        label_to_idx[class_dir] = idx
        idx_to_label[idx] = class_dir
    
    mapping_data = {
        'label_to_idx': label_to_idx,
        'idx_to_label': idx_to_label,
        'num_classes': len(class_dirs),
        'created_from': train_dir
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(mapping_data, f, indent=2)
    
    print(f"[INFO] ImageNet mapping created: {len(class_dirs)} classes -> {output_path}")
    return mapping_data


def count_classes(csv_path):
    """
    Quick helper function to count unique classes in a CSV without creating adapter.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        Integer number of unique classes in second column
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    return len(df.iloc[:, 1].unique())