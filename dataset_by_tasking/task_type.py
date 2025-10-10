from enum import Enum


class TaskType(Enum):
    IMAGE_CLASSIFICATION = "image_classification"
    TEXT_CLASSIFICATION = "text_classification" 
    TEXT_GENERATION = "text_generation"
    OBJECT_DETECTION = "object_detection"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"
    TABULAR_CLASSIFICATION = "tabular_classification"
    QUESTION_ANSWERING = "question_answering"