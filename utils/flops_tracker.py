from ptflops import get_model_complexity_info
import torch
import torchvision.models as models
from transformers import BertModel

def compute_flops_and_macs(model, input_shape, is_transformer=False):
    """
    Calcola FLOPs e MACs per un modello PyTorch.

    Args:
        model (torch.nn.Module): modello PyTorch (es. ResNet, Bert, ecc.)
        input_shape (tuple): shape del tensore di input (es. (3, 224, 224) per immagini, (seq_len,) per NLP)
        is_transformer (bool): se True, input shape verrà espanso a (1, seq_len)

    Returns:
        tuple: (FLOPs, MACs) in GigaOps (GFLOPs / GMACs)
    """
    model.eval()

    if is_transformer:
        # input_shape es. (128,) -> (1, 128)
        dummy_input = (1, input_shape[0])
        flops, params = get_model_complexity_info(model, dummy_input, as_strings=False,
                                                  print_per_layer_stat=False, verbose=False,
                                                  input_constructor=lambda _: {'input_ids': torch.ones(dummy_input, dtype=torch.long)})
    else:
        flops, params = get_model_complexity_info(model, input_shape, as_strings=False,
                                                  print_per_layer_stat=False, verbose=False)

    # 1 MAC = 2 FLOPs (approssimazione comune)
    macs = flops / 2

    return flops / 1e9, macs / 1e9

# Esempio 1: ResNet su immagini 224x224
resnet = models.resnet18()
flops, macs = compute_flops_and_macs(resnet, (3, 224, 224))
print(f"ResNet18: FLOPs={flops:.2f} GFLOPs, MACs={macs:.2f} GMACs")

# Esempio 2: BERT su input di 128 token
bert = BertModel.from_pretrained("bert-base-uncased")
flops, macs = compute_flops_and_macs(bert, (128,), is_transformer=True)
print(f"BERT-base: FLOPs={flops:.2f} GFLOPs, MACs={macs:.2f} GMACs")
