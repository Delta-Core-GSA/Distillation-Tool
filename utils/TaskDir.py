from utils.singleton import singleton

@singleton
class TaskDir:
    def __init__(self,labels=[], **kwargs):
        parts = []
        for key, value in kwargs.items():
            parts.append(f"{key.lower()}_{value}")
        if len(labels)>0:
            for label in labels:
                parts.append(f"_{label}")
        self.path = "_".join(parts)