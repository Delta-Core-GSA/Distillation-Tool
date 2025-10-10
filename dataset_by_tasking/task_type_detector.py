import pandas as pd
from dataset_by_tasking.task_type import *
from typing import Dict,Any

class TaskDetector:
    """Classe per rilevare automaticamente il tipo di task dal dataset"""
    
    @staticmethod
    def detect_task_type(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Rileva il tipo di task analizzando le colonne del DataFrame
        
        Returns:
            dict: Informazioni sul task rilevato
        """
        task_info = {
            'task_type': TaskType.TEXT_CLASSIFICATION,  # Default
            'input_columns': [],
            'target_columns': [],
            'num_classes': None,
            'is_multiclass': False,
            'data_type': 'unknown'
        }
        
        columns = df.columns.tolist()
        print(f"🔍 [TASK DETECTOR] Analizzando colonne: {columns}")
        
        # 1. Identificazione colonne di input e target
        common_input_cols = ['text', 'sentence', 'review', 'comment', 'image_path', 'image', 'input', 'question', 'context']
        common_target_cols = ['label', 'target', 'class', 'category', 'sentiment', 'score', 'answer']
        
        input_cols = [col for col in columns if col.lower() in common_input_cols]
        target_cols = [col for col in columns if col.lower() in common_target_cols]
        
        # Se non trova colonne standard, usa euristica
        if not input_cols:
            # Prima colonna probabilmente input, escluse quelle che sembrano target
            potential_inputs = [col for col in columns if col.lower() not in common_target_cols]
            input_cols = [potential_inputs[0]] if potential_inputs else [columns[0]]
        
        if not target_cols:
            # Ultima colonna probabilmente target, escluse quelle di input
            potential_targets = [col for col in columns if col not in input_cols]
            target_cols = [potential_targets[-1]] if potential_targets else [columns[-1]]
        
        task_info['input_columns'] = input_cols
        task_info['target_columns'] = target_cols
        
        print(f"📊 [TASK DETECTOR] Input columns: {input_cols}")
        print(f"🎯 [TASK DETECTOR] Target columns: {target_cols}")
        
        # 2. Determinazione del tipo di dati e task
        task_info['task_type'] = TaskDetector._determine_task_type(df, input_cols, target_cols)
        task_info['data_type'] = TaskDetector._determine_data_type(df, input_cols)
        
        # 3. Analisi target per classificazione
        if target_cols and 'classification' in task_info['task_type'].value:
            target_col = target_cols[0]
            unique_values = df[target_col].nunique()
            task_info['num_classes'] = unique_values
            task_info['is_multiclass'] = unique_values > 2
            
            print(f"📈 [TASK DETECTOR] Numero classi: {unique_values}")
        
        print(f"✅ [TASK DETECTOR] Task rilevato: {task_info['task_type'].value}")
        return task_info
    
    @staticmethod
    def _determine_task_type(df: pd.DataFrame, input_cols: list, target_cols: list) -> TaskType:
        """Determina il tipo di task specifico"""
        
        if not input_cols:
            return TaskType.TEXT_CLASSIFICATION
        
        input_col = input_cols[0]
        input_col_lower = input_col.lower()
        
        # Analisi basata sui nomi delle colonne
        if 'image' in input_col_lower or 'path' in input_col_lower:
            return TaskType.IMAGE_CLASSIFICATION
        
        elif 'question' in input_col_lower or 'context' in input_col_lower:
            return TaskType.QUESTION_ANSWERING
        
        elif any(keyword in input_col_lower for keyword in ['text', 'sentence', 'review', 'comment']):
            # Controlla se ha target per determinare classification vs generation
            if target_cols:
                target_col = target_cols[0]
                unique_values = df[target_col].nunique()
                # Se ha poche classi discrete, è classificazione
                if df[target_col].dtype in ['object', 'category'] or unique_values <= 50:
                    return TaskType.TEXT_CLASSIFICATION
                else:
                    return TaskType.TEXT_GENERATION
            else:
                return TaskType.TEXT_GENERATION
        
        else:
            # Analisi del contenuto per determinare il tipo
            sample_data = df[input_col].dropna().iloc[:10]
            
            # Controlla se sono path di file
            if sample_data.astype(str).str.contains(r'\.(jpg|jpeg|png|bmp|tiff?)$', case=False, regex=True).any():
                return TaskType.IMAGE_CLASSIFICATION
            
            # Controlla lunghezza media per distinguere testo da dati tabulari
            avg_length = sample_data.astype(str).str.len().mean()
            
            if avg_length > 20:  # Probabilmente testo
                if target_cols:
                    return TaskType.TEXT_CLASSIFICATION
                else:
                    return TaskType.TEXT_GENERATION
            else:
                # Dati numerici/tabulari
                return TaskType.TABULAR_CLASSIFICATION
    
    @staticmethod
    def _determine_data_type(df: pd.DataFrame, input_cols: list) -> str:
        """Determina il tipo di dati di input"""
        if not input_cols:
            return 'unknown'
        
        input_col = input_cols[0]
        input_col_lower = input_col.lower()
        
        if 'image' in input_col_lower or 'path' in input_col_lower:
            return 'image'
        elif any(keyword in input_col_lower for keyword in ['text', 'sentence', 'review', 'comment', 'question']):
            return 'text'
        else:
            # Analisi del contenuto
            sample_data = df[input_col].dropna().iloc[:5]
            avg_length = sample_data.astype(str).str.len().mean()
            
            if avg_length > 20:
                return 'text'
            elif sample_data.astype(str).str.contains(r'\.(jpg|jpeg|png|bmp)$', case=False, regex=True).any():
                return 'image'
            else:
                return 'numerical'

