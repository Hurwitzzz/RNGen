import logging
import os
from typing import Dict, Optional

import torch

logger = logging.getLogger(__name__)


def _save_weights(
    state_dict: Dict[str, torch.Tensor],
    save_directory: str,
    weights_description: str,
    safe_serialization: bool = True,
) -> Optional[str]:
    """Save model weights using safetensors when available."""
    if not state_dict:
        logger.info("No %s to save.", weights_description)
        return None

    os.makedirs(save_directory, exist_ok=True)
    file_path = None

    if safe_serialization:
        try:
            from safetensors.torch import save_file

            file_path = os.path.join(save_directory, "model.safetensors")
            save_file(state_dict, file_path)
            logger.info("Saved %s to %s", weights_description, file_path)
        except ImportError:
            logger.warning(
                "safetensors not available, falling back to PyTorch format for %s",
                weights_description,
            )
            file_path = os.path.join(save_directory, "pytorch_model.bin")
            torch.save(state_dict, file_path)
            logger.info("Saved %s to %s", weights_description, file_path)
        except Exception as e:
            logger.error(
                "Failed to save %s using safetensors: %s",
                weights_description,
                e,
            )
            file_path = os.path.join(save_directory, "pytorch_model.bin")
            torch.save(state_dict, file_path)
            logger.info("Saved %s to %s (fallback)", weights_description, file_path)
    else:
        file_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(state_dict, file_path)
        logger.info("Saved %s to %s", weights_description, file_path)

    return file_path


def _load_weights_from_path(path: str) -> Optional[Dict[str, torch.Tensor]]:
    """Load model weights from a checkpoint directory."""
    safetensors_path = os.path.join(path, "model.safetensors")
    pytorch_path = os.path.join(path, "pytorch_model.bin")
    state_dict = None

    if os.path.exists(safetensors_path):
        try:
            from safetensors.torch import load_file

            state_dict = load_file(safetensors_path, device="cpu")
        except ImportError:
            logger.warning("safetensors not installed, cannot load .safetensors file.")
        except Exception as e:
            logger.warning("Failed to load %s: %s", safetensors_path, e)
    elif os.path.exists(pytorch_path):
        try:
            state_dict = torch.load(pytorch_path, map_location="cpu")
        except Exception as e:
            logger.warning("Failed to load %s: %s", pytorch_path, e)

    return state_dict
