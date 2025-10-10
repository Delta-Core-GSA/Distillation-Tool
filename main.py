#!/usr/bin/env python3
"""
Main script per Knowledge Distillation Framework - VERSIONE AVANZATA
Utilizza la nuova architettura modulare con Factory Pattern e Dependency Injection
"""

import os
import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional

# Import della nuova architettura avanzata + bridge funzionale
from provider.distillation_component_factory import DistillationComponentFactory
from provider.distillation_config_builder import DistillationConfigBuilder
from distiller import DistillerBridge  # Bridge completo e funzionale
from config.config import DatasetConfig, ModelConfig, TaskConfig
from provider.task_type_provieder import TaskType

# Import per validazione  
try:
    from provider.distiller_bridge import ComponentValidator
except ImportError:
    # Fallback se ComponentValidator non disponibile
    class ComponentValidator:
        @staticmethod
        def validate_full_config(config):
            return {}  # No validation

# =================== CONFIGURAZIONI PREDEFINITE ===================

def get_default_config():
    """Configurazione di default per quick start"""
    return {
        'temperature': 3.0,
        'alpha': 0.8,
        'epochs': 3,
        'learning_rate': 1e-4,
        'batch_size': 16,
        'eval_every': 1
    }

def get_example_paths():
    """Esempi di percorsi per testing rapido"""
    return {
        'teacher_path': 'models/text/bert_teacher.pt',
        'student_path': 'models/text/distilbert_student.pt', 
        'dataset_path': 'datasets/SST2/train.csv',
        'output_path': 'saved_models/distillation',
        'tokenizer_name': 'bert-base-uncased'  # Solo per text tasks
    }

# =================== MODERN DISTILLATION FUNCTIONS ===================

def modern_simple_distillation(teacher_path: str, student_path: str, dataset_path: str, 
                              output_path: str, tokenizer_name: str = None, 
                              custom_config: Dict = None):
    """
    MODALITÀ SEMPLICE MODERNA: Usa la nuova architettura con factory pattern
    Mantiene la semplicità d'uso ma con l'architettura modulare sottostante
    """
    print("🚀 === MODALITÀ SEMPLICE MODERNA (Factory Pattern) ===")
    
    # Verifica che i file esistano
    required_files = [teacher_path, student_path, dataset_path]
    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File non trovato: {file_path}")
    
    # Configurazione base
    base_config = get_default_config()
    if custom_config:
        base_config.update(custom_config)
    
    print(f"📂 Teacher: {teacher_path}")
    print(f"📂 Student: {student_path}")
    print(f"📊 Dataset: {dataset_path}")
    print(f"💾 Output: {output_path}")
    print(f"⚙️  Config: {base_config}")
    
    # 1. Usa ConfigBuilder per creare configurazione strutturata
    print("\n🔧 Creazione configurazione strutturata...")
    
    builder = DistillationConfigBuilder()
    
    # Dataset config
    builder.with_dataset(
        csv_path=dataset_path,
        tokenizer_name=tokenizer_name,
        batch_size=base_config.get('batch_size', 16)
    )
    
    # Teacher config
    builder.with_teacher_model(
        model_path=teacher_path,
        device='cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else None
    )
    
    # Student config  
    builder.with_student_model(
        model_path=student_path,
        device='cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else None
    )
    
    # Task config - num_classes sarà rilevato automaticamente
    builder.with_task_config(
        num_classes=2,  # Placeholder, sarà aggiornato automaticamente
        temperature=base_config.get('temperature', 3.0),
        alpha=base_config.get('alpha', 0.8),
        learning_rate=base_config.get('learning_rate', 1e-4),
        epochs=base_config.get('epochs', 3)
    )
    
    # Build configurazione completa
    structured_config = builder.build()
    
    # 2. Crea Factory con dependency injection
    print("🏭 Inizializzazione Component Factory...")
    factory = DistillationComponentFactory()
    
    # 3. Converti configurazione strutturata per bridge funzionale
    print("🌉 Creazione DistillerBridge con architettura ibrida...")
    
    # Estrai parametri per il DistillerBridge originale
    dataset_config = structured_config['dataset']
    teacher_config = structured_config['teacher'] 
    student_config = structured_config['student']
    task_config = structured_config['task']
    
    # Converti TaskConfig in dizionario semplice
    simple_config = {
        'temperature': task_config.temperature,
        'alpha': task_config.alpha,
        'epochs': task_config.epochs,
        'learning_rate': task_config.learning_rate,
        'batch_size': dataset_config.batch_size
    }
    
    # Usa il DistillerBridge completo e funzionale
    bridge = DistillerBridge(
        teacher_path=teacher_config.model_path,
        student_path=student_config.model_path,
        dataset_path=dataset_config.csv_path,
        output_path=output_path,
        tokenizer_name=dataset_config.tokenizer_name,
        config=simple_config
    )
    
    # 4. Info sul setup (ora disponibili dalla factory)
    print(f"\n📈 Setup completato:")
    print(f"   - Task type: {bridge.dataset_info['task_type']}")
    print(f"   - Numero classi: {bridge.dataset_info['num_classes']}")
    print(f"   - Campioni dataset: {bridge.dataset_info['num_samples']}")
    
    # 5. Esegui distillazione con la nuova architettura
    print(f"\n🔥 Avvio distillazione con architettura modulare...")
    bridge.distill()
    
    print(f"✅ Distillazione completata! Output salvato in: {output_path}")
    return bridge

def advanced_modern_distillation(config_file: str):
    """
    MODALITÀ AVANZATA MODERNA: Configurazione completa con validazione
    Usa tutte le features della nuova architettura
    """
    print("🔧 === MODALITÀ AVANZATA MODERNA (Full Features) ===")
    
    # 1. Carica e valida configurazione
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"File di configurazione non trovato: {config_file}")
    
    with open(config_file, 'r') as f:
        raw_config = json.load(f)
    
    print(f"📄 Configurazione caricata da: {config_file}")
    
    # 2. Usa ConfigBuilder per parsing e validazione
    print("🔧 Parsing e validazione configurazione...")
    
    builder = DistillationConfigBuilder()
    
    # Dataset config
    dataset_cfg = raw_config['dataset']
    builder.with_dataset(
        csv_path=dataset_cfg['csv_path'],
        tokenizer_name=dataset_cfg.get('tokenizer_name'),
        max_samples=dataset_cfg.get('max_samples'),
        batch_size=dataset_cfg.get('batch_size', 16),
        imagenet_mapping_path=dataset_cfg.get('imagenet_mapping_path')
    )
    
    # Teacher model config
    teacher_cfg = raw_config['teacher']
    builder.with_teacher_model(
        model_path=teacher_cfg['model_path'],
        device=teacher_cfg.get('device'),
        load_in_8bit=teacher_cfg.get('load_in_8bit', False)
    )
    
    # Student model config  
    student_cfg = raw_config['student']
    builder.with_student_model(
        model_path=student_cfg['model_path'],
        device=student_cfg.get('device'),
        load_in_8bit=student_cfg.get('load_in_8bit', False)
    )
    
    # Task config
    task_cfg = raw_config['task']
    builder.with_task_config(
        num_classes=task_cfg.get('num_classes', 2),  # Sarà aggiornato automaticamente
        temperature=task_cfg.get('temperature', 3.0),
        alpha=task_cfg.get('alpha', 0.8),
        learning_rate=task_cfg.get('learning_rate', 1e-4),
        epochs=task_cfg.get('epochs', 3)
    )
    
    # Build configurazione strutturata
    structured_config = builder.build()
    
    # 3. Validazione configurazione con ComponentValidator
    print("✅ Validazione configurazione...")
    validation_errors = ComponentValidator.validate_full_config(structured_config)
    
    if validation_errors:
        print("❌ Errori di validazione trovati:")
        for component, errors in validation_errors.items():
            print(f"  {component}:")
            for error in errors:
                print(f"    - {error}")
        raise ValueError("Configurazione non valida")
    
    print("✅ Configurazione validata con successo!")
    
    # 4. Crea Factory con provider personalizzati se necessario
    print("🏭 Inizializzazione Advanced Component Factory...")
    
    # Puoi customizzare i provider qui se necessario
    from provider.providers import ModularDatasetProvider, StandardModelProvider
    
    factory = DistillationComponentFactory(
        dataset_provider=ModularDatasetProvider(),
        model_provider=StandardModelProvider()
    )
    
    # 5. Crea DistillerBridge con architettura ibrida
    print("🌉 Creazione DistillerBridge con configurazione avanzata...")
    
    # Estrai parametri per il bridge funzionale
    dataset_config = structured_config['dataset']
    teacher_config = structured_config['teacher'] 
    student_config = structured_config['student']
    task_config = structured_config['task']
    
    # Output path personalizzato
    output_path = raw_config.get('output_path', 'output/advanced_distilled')
    
    # Converti configurazione avanzata in formato semplice
    simple_config = {
        'temperature': task_config.temperature,
        'alpha': task_config.alpha,
        'epochs': task_config.epochs,
        'learning_rate': task_config.learning_rate,
        'batch_size': dataset_config.batch_size,
        'eval_every': raw_config.get('eval_every', 1)
    }
    
    # Usa DistillerBridge completo con configurazione preparata dalla factory
    bridge = DistillerBridge(
        teacher_path=teacher_config.model_path,
        student_path=student_config.model_path,
        dataset_path=dataset_config.csv_path,
        output_path=output_path,
        tokenizer_name=dataset_config.tokenizer_name,
        config=simple_config
    )
    
    # 6. Informazioni dettagliate sul setup
    print(f"\n📊 Setup Dettagliato:")
    print(f"   - Task type: {bridge.dataset_info['task_type']}")
    print(f"   - Numero classi: {bridge.dataset_info['num_classes']}")
    print(f"   - Campioni dataset: {bridge.dataset_info['num_samples']}")
    print(f"   - Task handler: {type(bridge.task_adapter).__name__}")
    
    # 7. Esegui distillazione avanzata
    print(f"\n🔥 Avvio distillazione avanzata...")
    bridge.distill()
    
    # 8. Salva metadati aggiuntivi
    metadata = {
        'experiment_name': raw_config.get('experiment_name', 'advanced_experiment'),
        'description': raw_config.get('description', ''),
        'config_file': config_file,
        'factory_components': {
            'dataset_provider': type(factory.dataset_provider).__name__,
            'model_provider': type(factory.model_provider).__name__
        }
    }
    
    metadata_path = os.path.join(output_path, 'experiment_metadata.json')
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print(f"✅ Distillazione avanzata completata!")
    print(f"📋 Metadati esperimento salvati in: {metadata_path}")
    
    return bridge

def create_modern_config(output_file: str = "modern_distillation_config.json"):
    """
    Crea un file di configurazione di esempio per la nuova architettura
    """
    modern_config = {
        "experiment_name": "bert_distillation_experiment",
        "description": "Distillazione BERT Large -> BERT Base con architettura modulare",
        
        "dataset": {
            "csv_path": "datasets/SST2/train.csv",
            "tokenizer_name": "bert-base-uncased",
            "max_samples": None,
            "batch_size": 32,
            "imagenet_mapping_path": None
        },
        
        "teacher": {
            "model_path": "models/text/bert_large_teacher.pt",
            "device": "cuda",
            "load_in_8bit": False
        },
        
        "student": {
            "model_path": "models/text/bert_base_student.pt", 
            "device": "cuda",
            "load_in_8bit": False
        },
        
        "task": {
            "num_classes": 2,
            "temperature": 4.0,
            "alpha": 0.7,
            "learning_rate": 2e-5,
            "epochs": 8
        },
        
        "output_path": "experiments/bert_distillation/run_001",
        
        "advanced_options": {
            "gradient_checkpointing": True,
            "mixed_precision": True,
            "custom_task_provider": None
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(modern_config, f, indent=2)
    
    print(f"📄 Configurazione moderna creata: {output_file}")
    print("🔧 Include tutte le features della nuova architettura")
    print("✏️  Personalizza secondo le tue necessità")

def interactive_modern_setup():
    """
    Setup interattivo per la nuova architettura
    Include validazione e rilevamento automatico
    """
    print("🎯 === SETUP INTERATTIVO MODERNO ===")
    print("Setup guidato con validazione automatica e rilevamento task\n")
    
    # 1. Raccolta input con validazione
    while True:
        teacher_path = input("📁 Percorso Teacher Model (.pt): ").strip()
        if os.path.exists(teacher_path):
            break
        print(f"❌ File non trovato: {teacher_path}")
    
    while True:
        student_path = input("📁 Percorso Student Model (.pt): ").strip()
        if os.path.exists(student_path):
            break
        print(f"❌ File non trovato: {student_path}")
    
    while True:
        dataset_path = input("📊 Percorso Dataset (.csv): ").strip()
        if os.path.exists(dataset_path):
            break
        print(f"❌ File non trovato: {dataset_path}")
    
    output_path = input("💾 Cartella Output (default: experiments/interactive/): ").strip() or "experiments/interactive/"
    
    # 2. Validazione automatica dei componenti
    print("\n🔍 Validazione automatica componenti...")
    
    # Valida dataset
    try:
        from provider.providers import ModularDatasetProvider
        dataset_provider = ModularDatasetProvider()
        
        # Quick check dataset
        import pandas as pd
        df = pd.read_csv(dataset_path)
        print(f"✅ Dataset: {df.shape[0]} campioni, {df.shape[1]} colonne")
        
        # Rileva task type automaticamente
        first_col = df.iloc[:, 0].astype(str)
        if first_col.str.len().mean() > 30:
            detected_task = "text"
            tokenizer_name = input("🔤 Nome Tokenizer (default: bert-base-uncased): ").strip() or "bert-base-uncased"
        elif first_col.str.contains(r'\.(jpg|jpeg|png)$', case=False).any():
            detected_task = "image"
            tokenizer_name = None
        else:
            detected_task = "tabular"
            tokenizer_name = None
        
        print(f"🎯 Task rilevato automaticamente: {detected_task}")
        
        # Rileva numero classi automaticamente
        if df.shape[1] > 1:
            num_classes = df.iloc[:, 1].nunique()
            print(f"🔢 Numero classi rilevate: {num_classes}")
        else:
            num_classes = 2  # Default
            
    except Exception as e:
        print(f"⚠️  Errore analisi dataset: {e}")
        tokenizer_name = None
        detected_task = "unknown"
        num_classes = 2
    
    # Valida modelli
    try:
        from provider.providers import StandardModelProvider
        model_provider = StandardModelProvider()
        
        # Quick validation (senza caricare completamente)
        if os.path.getsize(teacher_path) > 0:
            print("✅ Teacher model: file valido")
        if os.path.getsize(student_path) > 0:
            print("✅ Student model: file valido")
            
    except Exception as e:
        print(f"⚠️  Errore validazione modelli: {e}")
    
    # 3. Parametri di distillazione con suggerimenti intelligenti
    print(f"\n⚙️  Parametri di distillazione per task '{detected_task}':")
    
    # Suggerimenti basati sul task type
    if detected_task == "text":
        default_temp = 4.0
        default_alpha = 0.7
        default_lr = 2e-5
        default_epochs = 5
    elif detected_task == "image":
        default_temp = 3.5
        default_alpha = 0.8
        default_lr = 1e-4
        default_epochs = 10
    else:
        default_temp = 3.0
        default_alpha = 0.8
        default_lr = 1e-4
        default_epochs = 3
    
    temperature = float(input(f"🌡️  Temperature (default: {default_temp}): ") or str(default_temp))
    alpha = float(input(f"⚖️  Alpha (default: {default_alpha}): ") or str(default_alpha))
    learning_rate = float(input(f"📈 Learning Rate (default: {default_lr}): ") or str(default_lr))
    epochs = int(input(f"🔄 Epochs (default: {default_epochs}): ") or str(default_epochs))
    
    # 4. Crea configurazione usando il builder
    print(f"\n🔧 Creazione configurazione con ComponentFactory...")
    
    custom_config = {
        'temperature': temperature,
        'alpha': alpha,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'batch_size': 16  # Default conservativo
    }
    
    # 5. Usa la modalità semplice moderna
    print(f"\n🚀 Avvio distillazione moderna...")
    
    return modern_simple_distillation(
        teacher_path=teacher_path,
        student_path=student_path,
        dataset_path=dataset_path,
        output_path=output_path,
        tokenizer_name=tokenizer_name,
        custom_config=custom_config
    )

# =================== COMMAND LINE INTERFACE ===================

def parse_arguments():
    """Parser aggiornato per la nuova architettura"""
    parser = argparse.ArgumentParser(
        description="Knowledge Distillation Framework - Architettura Modulare",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🧠 Esempi di utilizzo con la NUOVA ARCHITETTURA:

# Modalità semplice moderna (con factory pattern)
python main.py --simple --teacher models/teacher.pt --student models/student.pt --dataset data/train.csv

# Modalità avanzata con validazione completa
python main.py --advanced --config modern_config.json

# Setup interattivo con rilevamento automatico
python main.py --interactive

# Crea config moderno con tutte le features
python main.py --create-config

# Test rapido con architettura moderna
python main.py --example

# Validazione setup completo
python main.py --validate-setup
        """
    )
    
    parser.add_argument('--simple', action='store_true', 
                       help='Modalità semplice con nuova architettura')
    parser.add_argument('--advanced', action='store_true',
                       help='Modalità avanzata con validazione completa')
    parser.add_argument('--interactive', action='store_true',
                       help='Setup interattivo con rilevamento automatico')
    parser.add_argument('--create-config', action='store_true',
                       help='Crea configurazione moderna di esempio')
    parser.add_argument('--example', action='store_true',
                       help='Test con architettura moderna')
    parser.add_argument('--validate-setup', action='store_true',
                       help='Valida setup e architettura')
    
    # Parametri per modalità semplice
    parser.add_argument('--teacher', type=str, help='Percorso teacher model')
    parser.add_argument('--student', type=str, help='Percorso student model')  
    parser.add_argument('--dataset', type=str, help='Percorso dataset CSV')
    parser.add_argument('--output', type=str, default='experiments/simple/',
                       help='Cartella output (default: experiments/simple/)')
    parser.add_argument('--tokenizer', type=str, 
                       help='Nome tokenizer per text tasks')
    
    # Parametri per modalità avanzata
    parser.add_argument('--config', type=str,
                       help='File di configurazione JSON moderna')
    
    # Parametri opzionali di distillazione
    parser.add_argument('--temperature', type=float, default=3.0,
                       help='Temperature per distillation (default: 3.0)')
    parser.add_argument('--alpha', type=float, default=0.8,
                       help='Alpha per loss weighting (default: 0.8)')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Numero di epochs (default: 3)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size (default: 16)')
    
    return parser.parse_args()

def validate_modern_setup():
    """
    Valida che la nuova architettura sia configurata correttamente
    """
    print("🔍 === VALIDAZIONE ARCHITETTURA MODERNA ===")
    
    validation_results = {}
    
    # Check import architettura moderna
    try:
        from provider.distillation_component_factory import DistillationComponentFactory
        validation_results['factory'] = True
        print("✅ DistillationComponentFactory importato")
    except ImportError as e:
        validation_results['factory'] = False
        print(f"❌ Errore import DistillationComponentFactory: {e}")
    
    try:
        from provider.distillation_config_builder import DistillationConfigBuilder
        validation_results['builder'] = True
        print("✅ DistillationConfigBuilder importato")
    except ImportError as e:
        validation_results['builder'] = False
        print(f"❌ Errore import DistillationConfigBuilder: {e}")
    
    try:
        from provider.distiller_bridge import DistillerBridge as AdvancedDistillerBridge
        validation_results['bridge'] = True
        print("✅ DistillerBridge avanzato importato")
    except ImportError as e:
        validation_results['bridge'] = False
        print(f"❌ Errore import DistillerBridge avanzato: {e}")
        
    # Test anche bridge funzionale
    try:
        from distiller import DistillerBridge as FunctionalDistillerBridge
        validation_results['functional_bridge'] = True
        print("✅ DistillerBridge funzionale importato")
    except ImportError as e:
        validation_results['functional_bridge'] = False
        print(f"❌ Errore import DistillerBridge funzionale: {e}")
    
    try:
        from provider.providers import ModularDatasetProvider, StandardModelProvider
        validation_results['providers'] = True
        print("✅ Provider moderni importati")
    except ImportError as e:
        validation_results['providers'] = False
        print(f"❌ Errore import Provider: {e}")
    
    # Test creazione factory
    if validation_results.get('factory', False):
        try:
            factory = DistillationComponentFactory()
            print("✅ Factory creato con successo")
            validation_results['factory_creation'] = True
        except Exception as e:
            print(f"❌ Errore creazione factory: {e}")
            validation_results['factory_creation'] = False
    
    # Test creazione builder
    if validation_results.get('builder', False):
        try:
            builder = DistillationConfigBuilder()
            print("✅ Builder creato con successo")
            validation_results['builder_creation'] = True
        except Exception as e:
            print(f"❌ Errore creazione builder: {e}")
            validation_results['builder_creation'] = False
    
    # Summary
    successful_components = sum(1 for v in validation_results.values() if v)
    total_components = len(validation_results)
    
    print(f"\n📊 Risultato validazione: {successful_components}/{total_components} componenti OK")
    
    if successful_components == total_components:
        print("🎉 Architettura moderna completamente funzionale!")
        return True
    else:
        print("⚠️  Alcuni componenti della nuova architettura non sono disponibili")
        print("💡 Controlla gli import e la struttura del progetto")
        return False

def main():
    """Funzione main aggiornata per la nuova architettura"""
    print("🧠 Knowledge Distillation Framework - Architettura Modulare")
    print("=" * 60)
    
    args = parse_arguments()
    
    try:
        # Validazione setup architettura moderna
        if args.validate_setup:
            validate_modern_setup()
            return 0
        
        # Modalità interattiva moderna
        if args.interactive:
            interactive_modern_setup()
        
        # Crea config moderno
        elif args.create_config:
            output_file = args.config or "modern_distillation_config.json"
            create_modern_config(output_file)
        
        # Modalità avanzata moderna
        elif args.advanced:
            if not args.config:
                raise ValueError("Modalità avanzata richiede --config file.json")
            advanced_modern_distillation(args.config)
        
        # Test con architettura moderna
        elif args.example:
            # Prima valida che l'architettura sia OK
            if not validate_modern_setup():
                print("❌ Architettura non validata, impossibile eseguire esempio")
                return 1
            
            paths = get_example_paths()
            print("⚠️  MODALITÀ ESEMPIO MODERNA - Assicurati che i file esistano:")
            for key, path in paths.items():
                print(f"   {key}: {path}")
            
            response = input("\nContinuare con architettura moderna? (y/N): ")
            if response.lower() == 'y':
                modern_simple_distillation(
                    teacher_path=paths['teacher_path'],
                    student_path=paths['student_path'],
                    dataset_path=paths['dataset_path'],
                    output_path=paths['output_path'],
                    tokenizer_name=paths['tokenizer_name']
                )
            else:
                print("Test annullato.")
        
        # Modalità semplice moderna
        elif args.simple:
            if not all([args.teacher, args.student, args.dataset]):
                raise ValueError("Modalità semplice richiede --teacher, --student, --dataset")
            
            custom_config = {
                'temperature': args.temperature,
                'alpha': args.alpha, 
                'epochs': args.epochs,
                'batch_size': args.batch_size
            }
            
            modern_simple_distillation(
                teacher_path=args.teacher,
                student_path=args.student,
                dataset_path=args.dataset,
                output_path=args.output,
                tokenizer_name=args.tokenizer,
                custom_config=custom_config
            )
        
        # Default: setup interattivo moderno
        else:
            print("Nessuna modalità specificata. Avvio setup interattivo moderno...\n")
            # Prima verifica che l'architettura sia OK
            if validate_modern_setup():
                print()
                interactive_modern_setup()
            else:
                print("❌ Impossibile procedere con architettura non validata")
                return 1
    
    except KeyboardInterrupt:
        print("\n❌ Processo interrotto dall'utente")
    except Exception as e:
        print(f"\n❌ Errore: {e}")
        print("\n💡 Usa --help per vedere tutte le opzioni disponibili")
        print("🔧 Usa --validate-setup per controllare la configurazione")
        return 1
    
    return 0

# =================== MODERN QUICK START FUNCTIONS ===================

def modern_quick_text_distillation(teacher_path: str, student_path: str, 
                                 dataset_path: str, output_path: str = "experiments/text/"):
    """Shortcut moderno per text classification distillation"""
    return modern_simple_distillation(
        teacher_path=teacher_path,
        student_path=student_path, 
        dataset_path=dataset_path,
        output_path=output_path,
        tokenizer_name="bert-base-uncased",
        custom_config={'temperature': 4.0, 'alpha': 0.7, 'learning_rate': 2e-5}
    )

def modern_quick_image_distillation(teacher_path: str, student_path: str,
                                  dataset_path: str, output_path: str = "experiments/image/"):
    """Shortcut moderno per image classification distillation"""
    return modern_simple_distillation(
        teacher_path=teacher_path,
        student_path=student_path,
        dataset_path=dataset_path, 
        output_path=output_path,
        tokenizer_name=None,
        custom_config={'temperature': 3.5, 'alpha': 0.8, 'epochs': 10}
    )

# =================== ENTRY POINT ===================

if __name__ == "__main__":
    exit(main())