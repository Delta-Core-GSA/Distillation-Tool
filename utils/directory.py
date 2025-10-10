import os
from datetime import datetime

class ProjectStructure:
    """
    Class that manages the creation of a predefined folder structure for model training and distillation.

    Structure created in the current working directory:
    saved_models/
    ├── pretrain/
    └── distillation/
        └── {modelname}_{label_value}/
            ├── teacher/
            ├── student/
            └── logs_{date}/
    """

    def __init__(self):
        """
        Initialize the structure. Creates the root and basic folders (saved_models/...) in the current working directory.
        """
        self.base_path = os.getcwd()
        self.saved_models_path = os.path.join(self.base_path, "saved_models")
        self.pretrain_path = os.path.join(self.saved_models_path, "pretrain")
        self.distillation_path = os.path.join(self.saved_models_path, "distillation")

        # Ensure base folders exist
        self._create_directory(self.saved_models_path)
        self._create_directory(self.pretrain_path)
        self._create_directory(self.distillation_path)

    def _create_directory(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def create_distillation_folder(self, model_name, label_value):
        """
        Creates a new distillation folder with the following structure:
        distillation/
        └── {modelname}_{label_value}/
            ├── teacher/
            ├── student/
            └── logs_{date}/

        :param model_name: Name of the model being distilled
        :param label_value: A string representing the experiment label (e.g., 'exp42')

        :return: Path to the student model file (e.g., saved_models/distillation/vitbase_exp1/student/student.pt)
        """
        model_folder_name = f"{model_name}_{label_value}"
        model_folder_path = os.path.join(self.distillation_path, model_folder_name)

        # Create subfolders
        teacher_path = os.path.join(model_folder_path, "teacher")
        student_path = os.path.join(model_folder_path, "student")
        logs_path = os.path.join(model_folder_path, f"logs_{datetime.now().strftime('%Y%m%d')}")

        for path in [teacher_path, student_path, logs_path]:
            self._create_directory(path)

        student_model_path = os.path.join(student_path)

        print(f"[INFO] Distillation folder structure created for '{model_name}' with label '{label_value}'.")
        return student_model_path
