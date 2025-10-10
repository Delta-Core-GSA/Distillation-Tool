import pandas as pd
from dataset_by_tasking.task_type import *
from typing import Dict, Any


class TaskDetector:
    """
    Automatic task type detector for machine learning datasets.
    Analyzes DataFrame structure and content to determine the appropriate task type.
    """
    
    @staticmethod
    def detect_task_type(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect task type by analyzing DataFrame columns and content.
        
        Args:
            df: Input DataFrame to analyze
            
        Returns:
            Dictionary containing:
                - task_type: Detected TaskType enum value
                - input_columns: List of detected input column names
                - target_columns: List of detected target column names
                - num_classes: Number of unique classes (for classification tasks)
                - is_multiclass: Boolean indicating if task has more than 2 classes
                - data_type: Type of input data (text, image, numerical, unknown)
        """
        task_info = {
            'task_type': TaskType.TEXT_CLASSIFICATION,
            'input_columns': [],
            'target_columns': [],
            'num_classes': None,
            'is_multiclass': False,
            'data_type': 'unknown'
        }
        
        columns = df.columns.tolist()
        print(f"[TASK DETECTOR] Analyzing columns: {columns}")
        
        # Step 1: Identify input and target columns
        common_input_cols = ['text', 'sentence', 'review', 'comment', 'image_path', 'image', 'input', 'question', 'context']
        common_target_cols = ['label', 'target', 'class', 'category', 'sentiment', 'score', 'answer']
        
        input_cols = [col for col in columns if col.lower() in common_input_cols]
        target_cols = [col for col in columns if col.lower() in common_target_cols]
        
        # Apply heuristics if standard columns not found
        if not input_cols:
            # First column is likely input, excluding target-like columns
            potential_inputs = [col for col in columns if col.lower() not in common_target_cols]
            input_cols = [potential_inputs[0]] if potential_inputs else [columns[0]]
        
        if not target_cols:
            # Last column is likely target, excluding input columns
            potential_targets = [col for col in columns if col not in input_cols]
            target_cols = [potential_targets[-1]] if potential_targets else [columns[-1]]
        
        task_info['input_columns'] = input_cols
        task_info['target_columns'] = target_cols
        
        print(f"[TASK DETECTOR] Input columns: {input_cols}")
        print(f"[TASK DETECTOR] Target columns: {target_cols}")
        
        # Step 2: Determine data type and task type
        task_info['task_type'] = TaskDetector._determine_task_type(df, input_cols, target_cols)
        task_info['data_type'] = TaskDetector._determine_data_type(df, input_cols)
        
        # Step 3: Analyze target for classification tasks
        if target_cols and 'classification' in task_info['task_type'].value:
            target_col = target_cols[0]
            unique_values = df[target_col].nunique()
            task_info['num_classes'] = unique_values
            task_info['is_multiclass'] = unique_values > 2
            
            print(f"[TASK DETECTOR] Number of classes: {unique_values}")
        
        print(f"[TASK DETECTOR] Detected task: {task_info['task_type'].value}")
        return task_info
    
    @staticmethod
    def _determine_task_type(df: pd.DataFrame, input_cols: list, target_cols: list) -> TaskType:
        """
        Determine specific task type based on column names and content.
        
        Args:
            df: DataFrame to analyze
            input_cols: List of identified input columns
            target_cols: List of identified target columns
            
        Returns:
            TaskType enum value representing the detected task
        """
        if not input_cols:
            return TaskType.TEXT_CLASSIFICATION
        
        input_col = input_cols[0]
        input_col_lower = input_col.lower()
        
        # Analysis based on column names
        if 'image' in input_col_lower or 'path' in input_col_lower:
            return TaskType.IMAGE_CLASSIFICATION
        
        elif 'question' in input_col_lower or 'context' in input_col_lower:
            return TaskType.QUESTION_ANSWERING
        
        elif any(keyword in input_col_lower for keyword in ['text', 'sentence', 'review', 'comment']):
            # Check target to determine classification vs generation
            if target_cols:
                target_col = target_cols[0]
                unique_values = df[target_col].nunique()
                # If has few discrete classes, it's classification
                if df[target_col].dtype in ['object', 'category'] or unique_values <= 50:
                    return TaskType.TEXT_CLASSIFICATION
                else:
                    return TaskType.TEXT_GENERATION
            else:
                return TaskType.TEXT_GENERATION
        
        else:
            # Content analysis to determine type
            sample_data = df[input_col].dropna().iloc[:10]
            
            # Check if file paths
            if sample_data.astype(str).str.contains(r'\.(jpg|jpeg|png|bmp|tiff?)$', case=False, regex=True).any():
                return TaskType.IMAGE_CLASSIFICATION
            
            # Check average length to distinguish text from tabular data
            avg_length = sample_data.astype(str).str.len().mean()
            
            if avg_length > 20:
                # Likely text data
                if target_cols:
                    return TaskType.TEXT_CLASSIFICATION
                else:
                    return TaskType.TEXT_GENERATION
            else:
                # Numerical/tabular data
                return TaskType.TABULAR_CLASSIFICATION
    
    @staticmethod
    def _determine_data_type(df: pd.DataFrame, input_cols: list) -> str:
        """
        Determine input data type (text, image, or numerical).
        
        Args:
            df: DataFrame to analyze
            input_cols: List of identified input columns
            
        Returns:
            String indicating data type: 'text', 'image', 'numerical', or 'unknown'
        """
        if not input_cols:
            return 'unknown'
        
        input_col = input_cols[0]
        input_col_lower = input_col.lower()
        
        # Column name-based detection
        if 'image' in input_col_lower or 'path' in input_col_lower:
            return 'image'
        elif any(keyword in input_col_lower for keyword in ['text', 'sentence', 'review', 'comment', 'question']):
            return 'text'
        else:
            # Content-based analysis
            sample_data = df[input_col].dropna().iloc[:5]
            avg_length = sample_data.astype(str).str.len().mean()
            
            if avg_length > 20:
                return 'text'
            elif sample_data.astype(str).str.contains(r'\.(jpg|jpeg|png|bmp)$', case=False, regex=True).any():
                return 'image'
            else:
                return 'numerical'