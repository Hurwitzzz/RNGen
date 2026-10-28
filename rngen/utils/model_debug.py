import logging

import torch


def log_model_parameter_stats(model: torch.nn.Module, logger: logging.Logger):
    """Log the number and percentage of trainable parameters in a model."""
    total_params = 0
    trainable_params = 0
    for _, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    percentage_trainable = (
        (trainable_params / total_params) * 100 if total_params > 0 else 0
    )

    logger.info("Model Parameter Statistics:")
    logger.info("  Total parameters: %s", f"{total_params:,}")
    logger.info("  Trainable parameters: %s", f"{trainable_params:,}")
    logger.info("  Percentage of trainable parameters: %.2f%%", percentage_trainable)

    for name, param in model.named_parameters():
        logger.debug(
            "Param: %s, Size: %s, Trainable: %s",
            name,
            param.numel(),
            param.requires_grad,
        )
