

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Set
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset_adapter.interfaces import IDatasetProcessor
from dataset_adapter.implementation import SmartTokenizerManager




class TextDatasetProcessor(IDatasetProcessor):
    """Processore specializzato per dataset di testo"""
    
    def __init__(self, tokenizer_manager: SmartTokenizerManager, batch_size: int = 16):
        self.tokenizer_manager = tokenizer_manager
        self.batch_size = batch_size
    
    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # Validazione specifica per text
        if df.iloc[:, 0].astype(str).str.len().mean() < 10:
            raise ValueError("Text column seems too short for text classification")
        return df
    
    def create_dataloader(self, processed_data: pd.DataFrame) -> DataLoader:
        from torch.utils.data import Dataset
        
        class TextDataset(Dataset):
            def __init__(self, df):
                self.texts = df.iloc[:, 0].astype(str).tolist()
                self.labels = df.iloc[:, 1].tolist()
            
            def __getitem__(self, idx):
                return self.texts[idx], self.labels[idx]
            
            def __len__(self):
                return len(self.texts)
        
        def collate_fn(batch):
            texts, labels = zip(*batch)
            
            tokenized = self.tokenizer_manager.tokenizer(
                list(texts),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            filtered = self.tokenizer_manager.filter_inputs(tokenized)
            filtered['labels'] = torch.tensor(labels)
            return filtered
        
        dataset = TextDataset(processed_data)
        return DataLoader(dataset, batch_size=self.batch_size, 
                         collate_fn=collate_fn, shuffle=True)

class ImageDatasetProcessor(IDatasetProcessor):
    """Processore specializzato per dataset di immagini"""
    
    def __init__(self, batch_size: int = 16):
        self.batch_size = batch_size
        self.transform = self._get_default_transform()
    
    def _get_default_transform(self):
        from torchvision import transforms
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # Validazione paths immagini
        paths = df.iloc[:, 0].astype(str)
        valid_extensions = paths.str.contains(r'\.(jpg|jpeg|png)$', case=False)
        valid_base64 = paths.str.startswith("data:image/")
        
        if not (valid_extensions.any() or valid_base64.any()):
            raise ValueError("No valid image paths or base64 found")
        
        return df
    
    def create_dataloader(self, processed_data: pd.DataFrame) -> DataLoader:
        from torch.utils.data import Dataset
        from PIL import Image
        import io, base64, os
        
        class ImageDataset(Dataset):
            def __init__(self, df, transform):
                self.paths = df.iloc[:, 0].tolist()
                self.labels = df.iloc[:, 1].tolist()
                self.transform = transform
            
            def __getitem__(self, idx):
                img_path = self.paths[idx]
                label = self.labels[idx]
                
                try:
                    if img_path.startswith("data:image"):
                        image = Image.open(io.BytesIO(
                            base64.b64decode(img_path.split(",")[1])
                        )).convert("RGB")
                    else:
                        image = Image.open(img_path).convert("RGB")
                except Exception:
                    image = Image.new('RGB', (224, 224), color='gray')
                
                if self.transform:
                    image = self.transform(image)
                
                return image, int(label)
            
            def __len__(self):
                return len(self.paths)
        
        dataset = ImageDataset(processed_data, self.transform)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                         num_workers=4, pin_memory=torch.cuda.is_available())