import torch
from pathlib import Path
from torch import nn


def save_model(model: nn.Module,
               target_dir: str,
               model_name:str):
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    assert model_name.endswith('.pth') or model_name.endswith('.pt'), 'model_name must end with .pth or .pt'
    save_path = target_dir / model_name

    torch.save(model.state_dict(),f=save_path)