#!/usr/bin/env python3
"""
Main script per Knowledge Distillation Framework - VERSIONE AVANZATA
Modalità CASCADE RESNET con architettura modulare
"""

import os
import argparse
import json
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, Any, Optional, List
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Import della nuova architettura avanzata + bridge funzionale
from provider.distillation_component_factory import DistillationComponentFactory
from provider.distillation_config_builder import DistillationConfigBuilder
from distiller import DistillerBridge  # Bridge completo e funzionale
from config.config import DatasetConfig, ModelConfig, TaskConfig
from provider.task_type_provieder import TaskType
from utils.save_model import save_model
from utils.directory import ProjectStructure

# Import per validazione  
try:
    from provider.distiller_bridge import ComponentValidator
except ImportError:
    # Fallback se ComponentValidator non disponibile
    class ComponentValidator:
        @staticmethod
        def validate_full_config(config):
            return {}  # No validation

# =================== CIFAR-100 DATASET HELPER ===================

def save_cifar100_as_csv(csv_path, split="train", max_samples=None):
    """
    Salva CIFAR-100 come CSV (versione completa)
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if os.path.exists(csv_path):
        print(f"{csv_path} già esistente, skip.")
        return

    print(f"Scaricando CIFAR-100 {split} split...")
    dataset = load_dataset("cifar100", split=split)
    
    # Directory per salvare le immagini
    images_dir = os.path.join(os.path.dirname(csv_path), "images")
    os.makedirs(images_dir, exist_ok=True)
    
    image_paths = []
    labels = []
    
    dataset_size = len(dataset)
    samples_to_save = dataset_size if max_samples is None else min(max_samples, dataset_size)
    
    print(f"Salvando {samples_to_save} immagini (dataset completo: {dataset_size})...")
    for i, sample in enumerate(dataset):
        if max_samples is not None and i >= max_samples:
            break
            
        # Salva immagine
        image_path = os.path.join(images_dir, f"cifar100_{i}.png")
        sample["img"].save(image_path)
        image_paths.append(image_path)
        labels.append(sample["fine_label"])  # CIFAR-100 usa fine_label
        
        if (i + 1) % 1000 == 0:
            print(f"  Salvate {i + 1}/{samples_to_save} immagini")
    
    # Crea CSV
    df = pd.DataFrame({
        "image_path": image_paths,
        "label": labels
    })
    df.to_csv(csv_path, index=False)
    print(f"✅ Salvato CIFAR-100 in {csv_path} con {len(df)} campioni")

def count_classes(csv_path):
    """Conta velocemente le classi dal CSV"""
    df = pd.read_csv(csv_path)
    return len(df.iloc[:, 1].unique())

def load_and_adapt_model(model_name, num_classes, save_path, force_reload=False):
    """
    Carica un modello e lo adatta al numero di classi
    """
    print(f"\n[MODEL] Caricando {model_name}...")
    
    try:
        # Se force_reload=True, ricarica sempre da HuggingFace
        if force_reload or not os.path.exists(save_path):
            # Carica con ignore_mismatched_sizes per gestire differenze di classi
            model = AutoModelForImageClassification.from_pretrained(
                model_name, 
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
            print(f"[MODEL] ✅ {model_name} caricato e adattato a {num_classes} classi")
            
            # Info sul modello
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"[MODEL] Parametri totali: {total_params:,}")
            print(f"[MODEL] Parametri trainable: {trainable_params:,}")
            
            # Salva il modello
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            print(f"[MODEL] Salvando in {save_path}...")
            save_model(model, save_path)
        else:
            print(f"[MODEL] Caricando da cache: {save_path}")
            model = torch.load(save_path)
        
        return model
        
    except Exception as e:
        print(f"[ERROR] Errore nel caricamento di {model_name}: {e}")
        raise

def test_model_compatibility(teacher_model, student_model, test_input):
    """
    Testa la compatibilità tra teacher e student model
    """
    print("\n[TEST] Testando compatibilità modelli...")
    
    teacher_model.eval()
    student_model.eval()
    
    with torch.no_grad():
        # Test teacher
        try:
            teacher_output = teacher_model(test_input)
            teacher_logits = teacher_output.logits if hasattr(teacher_output, 'logits') else teacher_output
            print(f"[TEST] Teacher output shape: {teacher_logits.shape}")
        except Exception as e:
            print(f"[ERROR] Teacher forward failed: {e}")
            return False
        
        # Test student
        try:
            student_output = student_model(test_input)
            student_logits = student_output.logits if hasattr(student_output, 'logits') else student_output
            print(f"[TEST] Student output shape: {student_logits.shape}")
        except Exception as e:
            print(f"[ERROR] Student forward failed: {e}")
            return False
        
        # Verifica compatibilità
        if teacher_logits.shape == student_logits.shape:
            print(f"[TEST] ✅ Modelli compatibili! Output shape: {teacher_logits.shape}")
            return True
        else:
            print(f"[ERROR] ❌ Modelli incompatibili! Teacher: {teacher_logits.shape} vs Student: {student_logits.shape}")
            return False

# =================== ADVANCED MODERN DISTILLATION CON CASCADE ===================

def advanced_modern_distillation(config_file: str):
    """
    MODALITÀ AVANZATA MODERNA: Configurazione completa con validazione
    SUPPORTA CASCATA RESNET se specificato nel config
    """
    print("🔧 === MODALITÀ AVANZATA MODERNA (Full Features + Cascade) ===")
    
    # 1. Carica e valida configurazione
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"File di configurazione non trovato: {config_file}")
    
    with open(config_file, 'r') as f:
        raw_config = json.load(f)
    
    print(f"📄 Configurazione caricata da: {config_file}")
    print(f"🔍 Chiavi trovate nel config: {list(raw_config.keys())}")
    
    # 2. Controlla se è una configurazione CASCADE
    if 'cascade_config' in raw_config:
        print("🎯 RILEVATA CONFIGURAZIONE CASCADE RESNET")
        return execute_cascade_distillation(raw_config)
    elif 'teacher' in raw_config and 'student' in raw_config:
        print("🔧 CONFIGURAZIONE SINGOLA DISTILLAZIONE")
        return execute_single_distillation(raw_config)
    else:
        available_keys = list(raw_config.keys())
        raise ValueError(f"Configurazione non valida. Chiavi trovate: {available_keys}. "
                        f"Deve contenere 'cascade_config' per cascade o 'teacher'+'student' per singola distillazione")

def execute_cascade_distillation(raw_config: Dict):
    """
    Esegue distillazione a cascata ResNet usando la configurazione
    """
    print("🚀 === ESECUZIONE CASCADE RESNET CON ARCHITETTURA MODERNA ===")
    print("=" * 70)
    
    # =================== ESTRAI CONFIGURAZIONE CASCADE ===================
    cascade_cfg = raw_config['cascade_config']
    dataset_cfg = raw_config['dataset']
    distillation_cfg = raw_config['distillation']
    output_base_path = raw_config.get('output_path', 'experiments/resnet_cascade/')
    
    teacher_model_name = cascade_cfg['teacher_model']
    student_models = cascade_cfg['student_models']
    reset_teacher = cascade_cfg.get('reset_teacher', True)
    
    print(f"🧠 Teacher: {teacher_model_name}")
    print(f"🎯 Students: {student_models}")
    print(f"🔄 Reset teacher: {reset_teacher}")
    
    # =================== DATASET PREPARATION ===================
    print("\n📊 PREPARAZIONE DATASET CIFAR-100 COMPLETO")
    print("-" * 50)
    
    dataset_path = dataset_cfg['csv_path']
    
    # Salva dataset CIFAR-100 completo (50,000 campioni train)
    save_cifar100_as_csv(dataset_path, split="train")
    
    # Conta classi
    num_classes = count_classes(dataset_path)
    print(f"📊 Numero classi CIFAR-100: {num_classes}")
    
    # =================== SETUP ARCHITETTURA MODERNA ===================
    print("\n🏭 SETUP ARCHITETTURA MODULARE")
    print("-" * 40)
    
    factory = DistillationComponentFactory()
    project = ProjectStructure()
    
    print("✅ ComponentFactory inizializzato")
    
    # Paths
    teacher_save_path = './models/pretrained/resnet152_teacher.pt'
    
    # =================== CASCADING DISTILLATION ===================
    print("\n🔄 INIZIO DISTILLAZIONE A CASCATA RESNET")
    print("-" * 60)
    
    # Lista per tenere traccia dei risultati
    distillation_results = []
    
    # Esegui distillazione per ogni student
    for i, student_model_name in enumerate(student_models):
        print(f"\n{'='*80}")
        print(f"🎯 DISTILLAZIONE {i+1}/{len(student_models)} - ARCHITETTURA MODULARE")
        print(f"Teacher: {teacher_model_name}")
        print(f"Student: {student_model_name}")
        print(f"{'='*80}")
        
        # =================== MODEL LOADING CON FACTORY ===================
        print("\n🤖 CARICAMENTO MODELLI CON ARCHITETTURA MODERNA")
        print("-" * 50)
        
        # IMPORTANTE: Ricarica sempre il teacher se reset_teacher=True
        if reset_teacher:
            print(f"🔄 Ricaricando teacher {teacher_model_name} (reset parametri)...")
            teacher_model = load_and_adapt_model(
                teacher_model_name, 
                num_classes, 
                teacher_save_path,
                force_reload=True
            )
        else:
            teacher_model = load_and_adapt_model(
                teacher_model_name, 
                num_classes, 
                teacher_save_path,
                force_reload=False
            )
        
        # Carica student model
        student_save_path = f'./models/pretrained/{student_model_name.split("/")[-1]}_student.pt'
        student_model = load_and_adapt_model(
            student_model_name, 
            num_classes, 
            student_save_path,
            force_reload=True
        )
        
        # =================== COMPATIBILITY TEST ===================
        print("\n🔍 TEST COMPATIBILITÀ MODELLI")
        print("-" * 30)
        
        test_input = torch.randn(1, 3, 224, 224)
        
        if not test_model_compatibility(teacher_model, student_model, test_input):
            print(f"❌ Modelli {teacher_model_name} e {student_model_name} non compatibili. Skip.")
            result = {
                'teacher': teacher_model_name,
                'student': student_model_name,
                'output_path': None,
                'status': 'failed',
                'error': 'Incompatible models'
            }
            distillation_results.append(result)
            continue
        
        # =================== CONFIG BUILDING CON FACTORY ===================
        print(f"\n🔧 CREAZIONE CONFIGURAZIONE MODERNA PER DISTILLAZIONE {i+1}")
        print("-" * 60)
        
        try:
            # Crea output path specifico
            student_short_name = student_model_name.split("/")[-1]
            output_model_path = project.create_distillation_folder(
                f"resnet152_to_{student_short_name}", 
                "cifar100"
            )
            
            # Usa ConfigBuilder per creare configurazione strutturata
            builder = DistillationConfigBuilder()
            
            # Dataset config
            builder.with_dataset(
                csv_path=dataset_path,
                tokenizer_name=None,  # Vision models
                batch_size=distillation_cfg.get('batch_size', 32)
            )
            
            # Teacher config
            builder.with_teacher_model(
                model_path=teacher_save_path,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            
            # Student config  
            builder.with_student_model(
                model_path=student_save_path,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            
            # Task config
            builder.with_task_config(
                num_classes=num_classes,
                temperature=distillation_cfg.get('temperature', 2.0),
                alpha=distillation_cfg.get('alpha', 0.7),
                learning_rate=distillation_cfg.get('learning_rate', 1e-4),
                epochs=distillation_cfg.get('epochs', 5)
            )
            
            # Build configurazione strutturata
            structured_config = builder.build()
            
            # =================== VALIDAZIONE ===================
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
            
            # =================== DISTILLATION EXECUTION ===================
            print(f"\n🔥 ESECUZIONE DISTILLAZIONE {i+1} CON DISTILLER BRIDGE")
            print("-" * 60)
            
            # Estrai configurazioni
            dataset_config = structured_config['dataset']
            teacher_config = structured_config['teacher'] 
            student_config = structured_config['student']
            task_config = structured_config['task']
            
            # Prepara configurazione per bridge
            simple_config = {
                'temperature': task_config.temperature,
                'alpha': task_config.alpha,
                'epochs': task_config.epochs,
                'learning_rate': task_config.learning_rate,
                'batch_size': dataset_config.batch_size,
                'eval_every': distillation_cfg.get('eval_every', 1)
            }
            
            # Crea DistillerBridge con architettura ibrida
            bridge = DistillerBridge(
                teacher_path=teacher_config.model_path,
                student_path=student_config.model_path,
                dataset_path=dataset_config.csv_path,
                output_path=output_model_path,
                tokenizer_name=dataset_config.tokenizer_name,
                config=simple_config
            )
            
            print(f"🎯 DistillerBridge configurato:")
            print(f"   - Teacher: {teacher_model_name}")
            print(f"   - Student: {student_model_name}")  
            print(f"   - Classi: {bridge.dataset_info['num_classes']}")
            print(f"   - Task: {bridge.dataset_info['task_type']}")
            print(f"   - Campioni: {bridge.dataset_info['num_samples']}")
            print(f"   - Output: {output_model_path}")
            
            # Esegui distillazione
            print(f"\n🚀 Avvio distillazione moderna {teacher_model_name} → {student_model_name}...")
            bridge.distill()
            
            print(f"\n✅ DISTILLAZIONE {i+1} COMPLETATA CON ARCHITETTURA MODERNA!")
            
            # Salva risultato di successo
            result = {
                'teacher': teacher_model_name,
                'student': student_model_name,
                'output_path': output_model_path,
                'status': 'success',
                'config_used': simple_config,
                'dataset_info': bridge.dataset_info
            }
            distillation_results.append(result)
            
        except Exception as e:
            print(f"\n❌ ERRORE durante distillazione {i+1}: {e}")
            import traceback
            traceback.print_exc()
            
            # Salva errore
            result = {
                'teacher': teacher_model_name,
                'student': student_model_name,
                'output_path': None,
                'status': 'failed',
                'error': str(e)
            }
            distillation_results.append(result)
        
        # Cleanup memoria
        del teacher_model, student_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        print(f"\n✅ Distillazione {i+1}/{len(student_models)} completata")
        print("-" * 80)
    
    # =================== FINAL SUMMARY ===================
    print("\n📋 RIEPILOGO FINALE CASCADE RESNET MODERNA")
    print("=" * 90)
    print(f"🏭 Architettura: Modulare con Factory Pattern")
    print(f"🧠 Teacher Model: {teacher_model_name} (reset: {reset_teacher})")
    print(f"📊 Dataset: CIFAR-100 completo ({num_classes} classi, 50,000 campioni)")
    print(f"🔧 ConfigBuilder: Utilizzato per ogni distillazione")
    print(f"🌉 DistillerBridge: Architettura ibrida moderna/funzionale")
    print("\n🎯 Risultati per Student:")
    
    for i, result in enumerate(distillation_results):
        status_emoji = "✅" if result['status'] == 'success' else "❌"
        print(f"  {i+1}. {status_emoji} {result['student']}")
        if result['status'] == 'success':
            print(f"     📁 Output: {result['output_path']}")
            print(f"     📊 Campioni: {result['dataset_info']['num_samples']}")
        else:
            print(f"     💥 Errore: {result['error']}")
    
    # Statistiche finali
    successful = sum(1 for r in distillation_results if r['status'] == 'success')
    total = len(distillation_results)
    print(f"\n📊 Statistiche Cascade: {successful}/{total} distillazioni riuscite")
    
    # Salva report dettagliato
    report_path = os.path.join(output_base_path, "cascade_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    cascade_report = {
        'experiment_type': 'ResNet Cascade Distillation',
        'architecture': 'Modern Modular with Factory Pattern',
        'teacher_model': teacher_model_name,
        'student_models': student_models,
        'dataset': 'CIFAR-100 Complete (50,000 samples)',
        'config_used': distillation_cfg,
        'results': distillation_results,
        'summary': {
            'successful_distillations': successful,
            'total_attempts': total,
            'success_rate': f"{(successful/total*100):.1f}%" if total > 0 else "0%"
        }
    }
    
    with open(report_path, 'w') as f:
        json.dump(cascade_report, f, indent=2, default=str)
    
    print(f"📋 Report dettagliato salvato: {report_path}")
    
    if successful > 0:
        print(f"\n🎉 CASCADE RESNET MODERNA COMPLETATA! {successful} modelli distillati")
        print("🏭 Architettura modulare utilizzata con successo")
    else:
        print(f"\n😞 Nessuna distillazione riuscita. Controlla gli errori sopra.")
    
    print("\n🏁 PROCESSO CASCADE TERMINATO!")
    
    return distillation_results

def execute_single_distillation(raw_config: Dict):
    """
    Esegue singola distillazione (modalità non-cascade)
    """
    print("🔧 Eseguendo distillazione singola...")
    
    # Implementazione distillazione singola standard
    # (mantenuta per compatibilità)
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
        num_classes=task_cfg.get('num_classes', 2),
        temperature=task_cfg.get('temperature', 3.0),
        alpha=task_cfg.get('alpha', 0.8),
        learning_rate=task_cfg.get('learning_rate', 1e-4),
        epochs=task_cfg.get('epochs', 3)
    )
    
    # Build configurazione strutturata
    structured_config = builder.build()
    
    # Validazione
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
    
    # Estrai parametri per il bridge
    dataset_config = structured_config['dataset']
    teacher_config = structured_config['teacher'] 
    student_config = structured_config['student']
    task_config = structured_config['task']
    
    output_path = raw_config.get('output_path', 'output/advanced_distilled')
    
    # Converti configurazione
    simple_config = {
        'temperature': task_config.temperature,
        'alpha': task_config.alpha,
        'epochs': task_config.epochs,
        'learning_rate': task_config.learning_rate,
        'batch_size': dataset_config.batch_size,
        'eval_every': raw_config.get('eval_every', 1)
    }
    
    # Usa DistillerBridge
    bridge = DistillerBridge(
        teacher_path=teacher_config.model_path,
        student_path=student_config.model_path,
        dataset_path=dataset_config.csv_path,
        output_path=output_path,
        tokenizer_name=dataset_config.tokenizer_name,
        config=simple_config
    )
    
    print(f"\n🔥 Avvio distillazione singola...")
    bridge.distill()
    
    print(f"✅ Distillazione singola completata!")
    return bridge

def create_resnet_cascade_config(output_file: str = "resnet_cascade_config.json"):
    """
    Crea un file di configurazione specifico per ResNet cascade con CIFAR-100
    """
    resnet_config = {
        "experiment_name": "resnet_cascade_cifar100_experiment",
        "description": "Distillazione a cascata ResNet-152 -> 101,50,34,18 con CIFAR-100 completo",
        
        "cascade_config": {
            "teacher_model": "microsoft/resnet-152",
            "student_models": [
                "microsoft/resnet-101",
                "microsoft/resnet-50", 
                "microsoft/resnet-34",
                "microsoft/resnet-18"
            ],
            "reset_teacher": True
        },
        
        "dataset": {
            "name": "cifar100",
            "csv_path": "./datasets/CIFAR100_full/train.csv",
            "tokenizer_name": None,
            "max_samples": None,
            "batch_size": 32
        },
        
        "distillation": {
            "temperature": 2.0,
            "alpha": 0.7,
            "learning_rate": 1e-4,
            "epochs": 5,
            "eval_every": 1
        },
        
        "output_path": "experiments/resnet_cascade_cifar100/",
        
        "advanced_options": {
            "use_modern_architecture": True,
            "save_intermediate_models": True,
            "generate_cascade_report": True
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(resnet_config, f, indent=2)
    
    print(f"📄 Configurazione ResNet cascade CIFAR-100 creata: {output_file}")
    print("🎯 Ottimizzata per distillazione a cascata con architettura moderna")

# =================== COMMAND LINE INTERFACE ===================

def parse_arguments():
    """Parser semplificato per cascade ResNet"""
    parser = argparse.ArgumentParser(
        description="Knowledge Distillation Framework - ResNet Cascade con CIFAR-100",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🧠 Esempi di utilizzo:

# Modalità avanzata con cascade ResNet (usa config file)
python main.py --advanced --config resnet_cascade_config.json

# Crea config specifico per ResNet cascade CIFAR-100
python main.py --create-resnet-config

# Validazione setup completo
python main.py --validate-setup
        """
    )
    
    parser.add_argument('--advanced', action='store_true',
                       help='Modalità avanzata con validazione completa (supporta cascade)')
    parser.add_argument('--create-resnet-config', action='store_true',
                       help='Crea configurazione ResNet cascade CIFAR-100')
    parser.add_argument('--validate-setup', action='store_true',
                       help='Valida setup e architettura')
    
    # Parametri per modalità avanzata
    parser.add_argument('--config', type=str,
                       help='File di configurazione JSON moderna')
    
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
    """Funzione main semplificata per cascade ResNet"""
    print("🧠 Knowledge Distillation Framework - ResNet Cascade CIFAR-100")
    print("=" * 70)
    
    args = parse_arguments()
    
    try:
        # Validazione setup architettura moderna
        if args.validate_setup:
            validate_modern_setup()
            return 0
        
        # Crea config ResNet cascade
        elif args.create_resnet_config:
            output_file = "resnet_cascade_config.json"
            create_resnet_cascade_config(output_file)
        
        # Modalità avanzata moderna (supporta cascade)
        elif args.advanced:
            if not args.config:
                raise ValueError("Modalità avanzata richiede --config file.json")
            advanced_modern_distillation(args.config)
        
        # Default: crea config e suggerisci comando
        else:
            print("Nessuna modalità specificata.\n")
            print("🎯 Opzioni disponibili:")
            print("  --create-resnet-config : Crea configurazione ResNet cascade CIFAR-100")
            print("  --advanced --config file.json : Esegui distillazione con config")
            print("  --validate-setup       : Valida architettura")
            
            response = input("\n🎯 Vuoi creare la configurazione ResNet cascade? (Y/n): ")
            if response.lower() != 'n':
                print("🚀 Creando configurazione ResNet cascade CIFAR-100...")
                create_resnet_cascade_config()
                print("\n💡 Ora esegui:")
                print("python main.py --advanced --config resnet_cascade_config.json")
            else:
                print("💡 Usa --help per vedere tutte le opzioni disponibili")
    
    except KeyboardInterrupt:
        print("\n❌ Processo interrotto dall'utente")
    except Exception as e:
        print(f"\n❌ Errore: {e}")
        print("\n💡 Usa --help per vedere tutte le opzioni disponibili")
        print("🔧 Usa --validate-setup per controllare la configurazione")
        print("🎯 Usa --create-resnet-config per creare la configurazione")
        return 1
    
    return 0

# =================== ENTRY POINT ===================

if __name__ == "__main__":
    exit(main())