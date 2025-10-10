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
    def __init__(self, csv_path, tokenizer_name=None, max_samples=None, imagenet_mapping_path=None):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.imagenet_mapping_path = imagenet_mapping_path
        
        # NUOVO: Set per tracciare input supportati e capacità tokenizer
        self.supported_inputs = None
        self.tokenizer_capabilities = None

        if max_samples:
            print(f"[INFO] Trimming dataset to max {max_samples} samples")
            self.df = self.df.head(max_samples)

        print(f"[INFO] Loaded dataset with shape: {self.df.shape}")
        self.mode = self.infer_mode()
        print(f"[INFO] Inferred mode: {self.mode}")
        
        # Inizializza tokenizer
        self.tokenizer = None
        if tokenizer_name:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            # NUOVO: Rileva capacità tokenizer dopo averlo caricato
            self.tokenizer_capabilities = self._detect_tokenizer_capabilities()

        # Gestione mapping label (ImageNet o generico)
        self.setup_label_mapping()

    def _detect_tokenizer_capabilities(self) -> Dict[str, bool]:
        """
        NUOVO: Rileva dinamicamente le capacità del tokenizer
        """
        if not self.tokenizer:
            return {}
            
        print(f"[DATASET] Rilevando capacità tokenizer...")
        
        test_texts = ["This is a test sentence."]
        capabilities = {}
        
        try:
            # Test tokenizzazione base
            base_output = self.tokenizer(
                test_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=64
            )
            
            # Verifica quali output sono generati
            for key in ['input_ids', 'attention_mask', 'token_type_ids', 'position_ids']:
                capabilities[key] = key in base_output
                if capabilities[key]:
                    print(f"[DATASET] ✅ Tokenizer genera: {key}")
            
            return capabilities
            
        except Exception as e:
            print(f"[DATASET] ❌ Errore rilevamento tokenizer: {e}")
            return {
                'input_ids': True,
                'attention_mask': True,
                'token_type_ids': False,
                'position_ids': False
            }
    
    def set_supported_inputs(self, supported_inputs: Set[str]):
        """
        NUOVO: Imposta quali input sono supportati dai modelli
        Chiamato dal TaskHandler dopo il rilevamento dinamico
        """
        self.supported_inputs = supported_inputs
        print(f"[DATASET] Input supportati impostati: {sorted(supported_inputs)}")

    def setup_label_mapping(self):
        """
        Configura il mapping delle label - può essere specifico per ImageNet o generico
        """
        if self.imagenet_mapping_path and os.path.exists(self.imagenet_mapping_path):
            print(f"[INFO] Caricamento mapping ImageNet da: {self.imagenet_mapping_path}")
            self.load_imagenet_mapping()
        else:
            print(f"[INFO] Creazione mapping automatico dalle label del dataset")
            self.create_automatic_mapping()
        
        # Applica il mapping al DataFrame
        self.apply_label_mapping()

    def get_dataset_info(self):
        """
        Ritorna informazioni essenziali del dataset
        """
        return {
            'num_classes': len(self.label_to_idx),
            'class_names': list(self.label_to_idx.keys()),
            'label_mapping': self.label_to_idx,
            'task_type': self.mode,  # "text", "image", "tabular"
            'num_samples': len(self.df)
        }

    def load_imagenet_mapping(self):
        """
        Carica il mapping ImageNet da file JSON
        """
        with open(self.imagenet_mapping_path, 'r') as f:
            mapping_data = json.load(f)
        
        self.label_to_idx = mapping_data.get('label_to_idx', {})
        self.idx_to_label = mapping_data.get('idx_to_label', {})
        # Converti le chiavi numeriche di idx_to_label in int
        self.idx_to_label = {int(k): v for k, v in self.idx_to_label.items()}
        
        print(f"[INFO] Mapping ImageNet caricato: {len(self.label_to_idx)} classi")

    def create_automatic_mapping(self):
        """
        Crea mapping automatico dalle label uniche nel dataset
        """
        unique_labels = self.df.iloc[:, 1].unique()
        self.label_to_idx = {}
        self.idx_to_label = {}
        
        for idx, label in enumerate(sorted(unique_labels)):
            self.label_to_idx[str(label)] = idx  # Assicura che sia stringa
            self.idx_to_label[idx] = str(label)
        
        print(f"[INFO] Mapping automatico creato: {len(unique_labels)} classi")

    def apply_label_mapping(self):
        """
        Applica il mapping delle label al DataFrame
        """
        # Converte le label in stringhe per il mapping
        self.df.iloc[:, 1] = self.df.iloc[:, 1].astype(str)
        
        # Applica il mapping
        original_labels = self.df.iloc[:, 1].copy()
        mapped_labels = self.df.iloc[:, 1].map(self.label_to_idx)
        
        # Controlla se ci sono label non mappate
        unmapped_mask = mapped_labels.isna()
        if unmapped_mask.any():
            unmapped_labels = original_labels[unmapped_mask].unique()
            print(f"[WARNING] Label non mappate trovate: {unmapped_labels[:10]}...")  # Mostra prime 10
            # Assegna 0 alle label non mappate (o potresti voler gestire diversamente)
            mapped_labels = mapped_labels.fillna(0)
        
        self.df.iloc[:, 1] = mapped_labels.astype(int)
        print(f"[INFO] Label mappate con successo. Range: {mapped_labels.min()}-{mapped_labels.max()}")

    def save_mapping(self, output_path):
        """
        Salva il mapping corrente in un file JSON
        """
        mapping_data = {
            'label_to_idx': self.label_to_idx,
            'idx_to_label': self.idx_to_label,
            'num_classes': len(self.label_to_idx)
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(mapping_data, f, indent=2)
        
        print(f"[INFO] Mapping salvato in: {output_path}")

    def infer_mode(self):
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
        if self.mode == "image":
            return self.get_image_loader()
        elif self.mode == "text":
            return self.get_text_loader()
        else:
            return self.get_tabular_loader()

    def get_image_loader(self):
        # Trasformazioni standard per ImageNet
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
                        # Gestione base64
                        image = Image.open(io.BytesIO(base64.b64decode(img_path.split(",")[1]))).convert("RGB")
                    elif os.path.exists(img_path):
                        # Gestione file path
                        image = Image.open(img_path).convert("RGB")
                    else:
                        print(f"[WARNING] File non trovato: {img_path}")
                        image = Image.new('RGB', (224, 224), color='gray')
                except Exception as e:
                    print(f"[ERROR] Errore caricamento immagine {img_path}: {e}")
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
        AGGIORNATO: Text loader con filtraggio dinamico degli input
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
            
            # Tokenizza con tutti i parametri possibili
            tokenized = self.tokenizer(
                list(texts),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
                add_special_tokens=True
            )
            
            # NUOVO: Filtraggio dinamico basato su input supportati
            if self.supported_inputs is not None:
                filtered_tokenized = self._filter_tokenizer_output(tokenized)
            else:
                # Fallback: usa strategia conservativa
                filtered_tokenized = self._conservative_filter(tokenized)
            
            # Aggiungi labels
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
        NUOVO: Filtra l'output del tokenizer basandosi sugli input supportati
        """
        filtered_output = {}
        '''
        print(f"[DATASET] Filtraggio output tokenizer...")
        print(f"[DATASET] Output tokenizer: {list(tokenized_output.keys())}")
        print(f"[DATASET] Input supportati: {sorted(self.supported_inputs)}")
        '''

        for key, tensor in tokenized_output.items():
            if key in self.supported_inputs:
                filtered_output[key] = tensor
                #print(f"[DATASET] ✅ Mantenuto: {key}")
            #else:
                #print(f"[DATASET] ❌ Rimosso: {key} (non supportato)")
        
        # Verifica che ci siano almeno gli input essenziali
        essential_inputs = ['input_ids', 'attention_mask']
        for essential in essential_inputs:
            if essential not in filtered_output and essential in tokenized_output:
                #print(f"[DATASET] ⚠️ Aggiungendo input essenziale: {essential}")
                filtered_output[essential] = tokenized_output[essential]
        
        return filtered_output
    
    def _conservative_filter(self, tokenized_output: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        NUOVO: Strategia conservativa quando non si conoscono gli input supportati
        Usa solo input universalmente supportati
        """
        print(f"[DATASET] Usando filtro conservativo...")
        
        # Input che funzionano con la maggior parte dei modelli
        safe_inputs = ['input_ids', 'attention_mask']
        
        filtered_output = {}
        for key in safe_inputs:
            if key in tokenized_output:
                filtered_output[key] = tokenized_output[key]
                print(f"[DATASET] ✅ Input sicuro incluso: {key}")
        
        # Controlla se possiamo includere token_type_ids basandoci sul tokenizer
        if (self.tokenizer_capabilities and 
            self.tokenizer_capabilities.get('token_type_ids', False) and
            'token_type_ids' in tokenized_output):
            
            # Test rapido per vedere se il tokenizer supporta davvero token_type_ids
            if self._test_token_type_ids_support():
                filtered_output['token_type_ids'] = tokenized_output['token_type_ids']
                print(f"[DATASET] ✅ token_type_ids incluso (testato)")
            else:
                print(f"[DATASET] ❌ token_type_ids escluso (test fallito)")
        
        return filtered_output
    
    def _test_token_type_ids_support(self) -> bool:
        """
        NUOVO: Testa se il tokenizer genera realmente token_type_ids validi
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
                # Verifica che non siano tutti zeri (che indicherebbe un placeholder)
                token_type_ids = test_output['token_type_ids']
                
                # Per BERT-like models, potrebbero essere tutti 0 per singole frasi
                # Ma dovrebbero esistere come tensor valido
                is_valid_shape = token_type_ids.shape == test_output['input_ids'].shape
                
                return is_valid_shape
            
            return False
            
        except Exception as e:
            print(f"[DATASET] Test token_type_ids fallito: {e}")
            return False

    def get_tabular_loader(self):
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
        DataLoader specifico per text generation task
        Assume che il dataset abbia input text nella prima colonna e target text nella seconda
        """
        if not self.tokenizer:
            raise ValueError("Tokenizer is required for generation mode.")

        class GenerationDataset(Dataset):
            def __init__(self, df):
                # Assumiamo che il dataset abbia input e target text
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
        Ritorna il numero di classi
        """
        return len(self.label_to_idx)

    def print_mapping_info(self):
        """
        Stampa informazioni sul mapping delle classi
        """
        print(f"[INFO] === MAPPING INFO ===")
        print(f"Numero classi: {len(self.label_to_idx)}")
        print(f"Range indici: 0 - {len(self.label_to_idx) - 1}")
        
        # Mostra alcuni esempi di mapping
        sample_size = min(10, len(self.label_to_idx))
        sample_items = list(self.label_to_idx.items())[:sample_size]
        print(f"Esempi mapping (primi {sample_size}):")
        for label, idx in sample_items:
            print(f"  '{label}' -> {idx}")
        print(f"[INFO] ====================")

    # =================== METODI DEBUG E DIAGNOSTICA ===================
    
    def debug_tokenizer_compatibility(self, sample_texts: List[str] = None):
        """
        NUOVO: Debug completo della compatibilità tokenizer
        """
        if not self.tokenizer:
            print("❌ Nessun tokenizer disponibile")
            return
        
        if sample_texts is None:
            sample_texts = [
                "This is a short test.",
                "This is a longer test sentence with more words for testing purposes.",
                "Another example text."
            ]
        
        print(f"\n🔍 DEBUG TOKENIZER COMPATIBILITY")
        print(f"Tokenizer: {getattr(self.tokenizer, 'name_or_path', 'Unknown')}")
        print("-" * 50)
        
        try:
            # Test con parametri diversi
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
                    
                    print(f"  ✅ Output keys: {list(output.keys())}")
                    for key, tensor in output.items():
                        print(f"    {key}: {tensor.shape}")
                        
                except Exception as e:
                    print(f"  ❌ Config failed: {e}")
        
        except Exception as e:
            print(f"❌ Debug failed: {e}")
    
    def get_tokenizer_report(self) -> Dict[str, Any]:
        """
        NUOVO: Report completo del tokenizer
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


# ===== FUNZIONI DI UTILITÀ PER IMAGENET =====

def create_imagenet_mapping_from_train_dir(train_dir, output_path):
    """
    Crea il mapping ImageNet dalle cartelle di training
    """
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Directory train non trovata: {train_dir}")
    
    class_dirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    class_dirs.sort()  # Ordinamento consistente
    
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
    
    print(f"[INFO] Mapping ImageNet creato: {len(class_dirs)} classi -> {output_path}")
    return mapping_data


def count_classes(csv_path):
    """
    Funzione helper veloce per contare classi senza creare adapter
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    return len(df.iloc[:, 1].unique())