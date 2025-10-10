import torch
from transformers import AutoModel,BatchEncoding
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class BaseModelAdapter:
    def __init__(self, model_path):
        #TODO CAMBIARE QUESTA RIGA DEL DEVICE
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.load(model_path, weights_only=False).to(self.device)
        self.model.eval()




#TODO ANDREA E: capire bene questo blocco e cercare di generalizzarlo per i vari modelli. Bisogna capire come gestiscono gli output i vari modelli
    def predict_logits(self, batch):
        # Verifica se il batch è un dizionario o un BatchEncoding
        if isinstance(batch, (dict, BatchEncoding)):
            output = self.model(**batch)  # Hugging Face models
        else:
            batch = batch.to(self.device)
            output = self.model(batch)    # e.g. image models

        # Restituzione dei logits
        if isinstance(output, dict):
            return output["logits"]
        elif hasattr(output, "logits"):
            return output.logits
        elif isinstance(output, torch.Tensor):
            return output
        else:
            raise ValueError(f"[ModelAdapter] Output non riconosciuto: {type(output)}")
        


    
    


