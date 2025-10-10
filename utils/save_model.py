import os
import torch




def save_model_to_pt(model,output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(model,output_path)
    print("Conversione del modello in pt avvenuta")





import os

def save_model(model, output_path):
    # Assicura che la directory padre esista
    dir_name = os.path.dirname(output_path)
    os.makedirs(dir_name, exist_ok=True)

    # Debug: stampiamo cosa stai cercando di salvare
    if not output_path.endswith(".pt"):
        raise ValueError(f"❌ Output path non valido: {output_path} (manca .pt?)")

    print(f"✅ Salvataggio modello in: {output_path}")
    torch.save(model, output_path)
