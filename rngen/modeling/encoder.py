import os
import torch
import torch.nn as nn
from typing import Optional, List, Tuple
from musicbert_hf.models import (
    CompoundEmbeddings,
    MusicBertConfig,
    MusicBertEncoder,
    MusicBert,
    freeze_layers,
)
from transformers.models.bert.modeling_bert import BertEmbeddings, BertPreTrainedModel
from rngen.modeling.io import _load_weights_from_path
from rngen.event_vocabulary import DEFAULT_SPELLING_TOKENS, event_vocab_size

import logging

logger = logging.getLogger(__name__)


def scale_spelling_downsampling_slice(model: nn.Module, scale: float = 1.0) -> None:
    """Scale the ninth Octuple+ downsampling slice in-place.

    This supports transfer ablations after a spelled MusicBERT checkpoint is
    loaded: 1.0 keeps the learned spelling path, 0.0 zeros it, and intermediate
    values keep a scaled version of the learned injection direction.
    """
    if scale is None or scale == 1.0:
        return

    embeddings = model.bert.embeddings
    compound_ratio = getattr(embeddings, "compound_ratio", None)
    if compound_ratio != 9:
        raise ValueError(
            "spelling_downsampling_scale is only supported for 9-token "
            f"SpelledMusicBert embeddings, got compound_ratio={compound_ratio}."
        )

    downsampling = embeddings.downsampling[0]
    hidden_size = model.config.hidden_size
    spelling_start = (compound_ratio - 1) * hidden_size
    spelling_end = compound_ratio * hidden_size
    with torch.no_grad():
        downsampling.weight[:, spelling_start:spelling_end].mul_(scale)
    logger.info("Scaled spelling downsampling slice by %s.", scale)


class SpelledCompoundEmbeddings(CompoundEmbeddings):
    """Extended CompoundEmbeddings that supports 9-token format with spelled pitch"""

    def __init__(
        self,
        config,
        *,
        compound_ratio: int = 9,
        upsample: bool = False,
        original_embeddings: CompoundEmbeddings = None,
        vocab_size: Optional[int] = None,
    ):
        """
        Initialize spelled compound embeddings with extended vocabulary.

        Args:
            config: Model configuration
            compound_ratio: Number of tokens per note (9 for spelled pitch)
            upsample: Whether to upsample hidden states in forward pass
            original_embeddings: Original embeddings to initialize from
            vocab_size: Size of vocabulary. Defaults to `config.vocab_size`.
        """
        self.compound_ratio = compound_ratio
        self.upsample = upsample

        if vocab_size is None:
            vocab_size = config.vocab_size

        # Update config vocab size for new tokens
        config.vocab_size = vocab_size

        # Call parent's parent (BertEmbeddings) to avoid CompoundEmbeddings.__init__
        BertEmbeddings.__init__(self, config)

        self.downsampling = nn.Sequential(
            nn.Linear(config.hidden_size * compound_ratio, config.hidden_size)
        )

        # Overwrite `position_embeddings` because we need to set padding_idx to match
        #   fairseq implementation
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=1
        )
        if hasattr(self, "token_type_ids"):
            delattr(self, "token_type_ids")

        # Initialize from original embeddings if provided
        if original_embeddings is not None:
            self._initialize_from_original(original_embeddings, config)

        self.config = config

    def _initialize_from_original(
        self, original_embeddings: CompoundEmbeddings, config
    ) -> None:
        """Initialize from original 8-token embeddings while preserving behavior"""
        with torch.no_grad():
            # Copy position embeddings
            self.position_embeddings.weight.copy_(
                original_embeddings.position_embeddings.weight
            )
            self.LayerNorm.weight.copy_(original_embeddings.LayerNorm.weight)
            self.LayerNorm.bias.copy_(original_embeddings.LayerNorm.bias)

            # Copy token type embeddings if they exist
            if hasattr(self, "token_type_embeddings") and hasattr(
                original_embeddings, "token_type_embeddings"
            ):
                self.token_type_embeddings.weight.copy_(
                    original_embeddings.token_type_embeddings.weight
                )

            # Handle vocabulary expansion: preserve original tokens, add spelling tokens.
            original_vocab_size = original_embeddings.word_embeddings.num_embeddings

            # Copy original word embeddings
            self.word_embeddings.weight[:original_vocab_size].copy_(
                original_embeddings.word_embeddings.weight
            )

            # Initialize new tokens with a random normal distribution matching
            # the statistics of existing embeddings.
            existing_mean = original_embeddings.word_embeddings.weight.mean()
            existing_std = original_embeddings.word_embeddings.weight.std()

            nn.init.normal_(
                self.word_embeddings.weight[original_vocab_size:],
                mean=existing_mean.item(),
                std=existing_std.item(),
            )

            # Initialize new downsampling to preserve original 8-token behavior
            old_weight = original_embeddings.downsampling[0].weight  # [768, 8*768]
            old_bias = original_embeddings.downsampling[0].bias  # [768]

            new_weight = self.downsampling[0].weight  # [768, 9*768]
            new_bias = self.downsampling[0].bias  # [768]

            new_weight.zero_()

            # Copy original weights for first 8 token positions
            for i in range(8):
                old_start = i * config.hidden_size
                old_end = (i + 1) * config.hidden_size
                new_start = i * config.hidden_size
                new_end = (i + 1) * config.hidden_size
                new_weight[:, new_start:new_end] = old_weight[:, old_start:old_end]

            new_bias.copy_(old_bias)

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        past_key_values_length: int = 0,
    ):
        """
        Forward pass for 9-token compound embeddings.
        `inputs_embeds` CANNOT be removed.
        """
        assert input_ids is not None and input_ids.ndim == 2

        batch, compound_seq = input_ids.shape
        ratio = self.compound_ratio

        assert (
            compound_seq % ratio == 0
        ), f"Sequence length should be multiple of {ratio} for compound mode"
        assert (
            not past_key_values_length
        ), "past_key_values_length not supported for compound mode"
        assert (
            position_ids is None
        ), "custom position_ids are not supported for compound mode"

        seq = compound_seq // ratio

        # Position IDs (fairseq begins from 2)
        position_ids = self.position_ids[:, 2 : seq + 2]

        # Get embeddings for all tokens
        inputs_embeds = self.word_embeddings(
            input_ids
        )  # [batch, compound_seq, hidden_size]

        # Reshape to group tokens
        inputs_embeds = inputs_embeds.view(batch, seq, ratio * self.config.hidden_size)

        # Downsample from 9*hidden_size to hidden_size
        inputs_embeds = self.downsampling(inputs_embeds)

        # Add position embeddings
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = inputs_embeds + position_embeddings

        # Apply layer norm and dropout
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class SpelledMusicBertConfig(MusicBertConfig):
    """Extended MusicBertConfig with spelled pitch support"""

    def __init__(self, *args, **kwargs):
        # Set defaults for spelled pitch. Callers that know the concrete event
        # vocabulary should pass `vocab_size` explicitly.
        if "vocab_size" not in kwargs:
            kwargs["vocab_size"] = event_vocab_size()
        kwargs.setdefault("compound_ratio", 9)
        kwargs.setdefault("tie_word_embeddings", False)

        super().__init__(*args, **kwargs)
        self.compound_ratio = kwargs.get("compound_ratio", 9)


class SpelledMusicBertEncoder(MusicBertEncoder):
    """Extended MusicBertEncoder with spelled pitch support"""

    def __init__(
        self,
        config,
        *,
        add_pooling_layer=False,
        upsample: bool = True,
        original_encoder: MusicBertEncoder = None,
    ):

        # super(MusicBertEncoder, self).__init__(config)
        from transformers import PreTrainedModel

        PreTrainedModel.__init__(self, config)

        self.config = config
        self.compound_ratio = config.compound_ratio
        self.attn_implementation = self.config._attn_implementation
        self.position_embedding_type = self.config.position_embedding_type
        self.pooler = None

        # Create spelled embeddings
        if original_encoder is not None:
            self.embeddings = SpelledCompoundEmbeddings(
                config,
                compound_ratio=self.compound_ratio,
                upsample=upsample,
                original_embeddings=original_encoder.embeddings,
            )
            self.encoder = original_encoder.encoder
            if hasattr(original_encoder, "pooler"):
                self.pooler = original_encoder.pooler
        else:
            self.embeddings = SpelledCompoundEmbeddings(
                config, compound_ratio=config.compound_ratio, upsample=upsample
            )
            from transformers.models.bert.modeling_bert import BertEncoder

            self.encoder = BertEncoder(config)
            if add_pooling_layer:
                from transformers.models.bert.modeling_bert import BertPooler

                self.pooler = BertPooler(config)

            self.post_init()  # Warning: Should NOT be called if we are initializing from an original model!!

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        """Forward pass with 9-token support"""
        if (
            attention_mask is None and input_ids is not None
        ):  # In our case, `attention_mask` is never None.
            batch_size, compound_seq = input_ids.shape
            assert (
                compound_seq % self.compound_ratio == 0
            ), f"Sequence length must be divisible by {self.compound_ratio}"
            seq = compound_seq // self.compound_ratio
            attention_mask = torch.ones((batch_size, seq), device=input_ids.device)

        return super(MusicBertEncoder, self).forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **{k: v for k, v in kwargs.items() if k not in ["upsample"]},
        )


class MusicBertBackbone(BertPreTrainedModel):
    """MusicBERT hidden states without a masked-language-model prediction head."""

    config_class = MusicBertConfig
    base_model_prefix = "bert"
    # Older encoder packages include an unused prediction head. Encoder tensor
    # mismatches must still fail rather than being silently reinitialized.
    _keys_to_ignore_on_load_unexpected = [r"^cls\."]

    def __init__(self, config, encoder=None):
        super().__init__(config)
        self.bert = encoder

    def get_input_embeddings(self):
        return self.bert.embeddings.word_embeddings

    def set_input_embeddings(self, embeddings):
        self.bert.embeddings.word_embeddings = embeddings

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        return self.bert(input_ids=input_ids, attention_mask=attention_mask, **kwargs)


class SpelledMusicBert(MusicBertBackbone):
    """MusicBERT backbone with nine-token spelled pitch inputs."""

    config_class = SpelledMusicBertConfig

    def __init__(self, config, original_model: MusicBert = None):
        super().__init__(config)

        # Create extended encoder
        if original_model is not None:
            self.bert = SpelledMusicBertEncoder(
                config,
                original_encoder=original_model.bert,
            )
        else:
            self.bert = SpelledMusicBertEncoder(config)
            # Initialize weights
            self.post_init()  # Warning: Should NOT be called if we are initializing from an original model!!

    @classmethod
    def from_original_musicbert(
        cls,
        original_model: MusicBert,
        spelling_vocab_size: Optional[int] = None,
        vocab_size: Optional[int] = None,
    ):
        """Create SpelledMusicBert from original MusicBert"""

        # Create new config
        original_config = original_model.config
        if vocab_size is not None:
            target_vocab_size = vocab_size
        elif spelling_vocab_size is not None:
            target_vocab_size = original_config.vocab_size + spelling_vocab_size
        else:
            raise ValueError(
                "Either vocab_size or spelling_vocab_size must be provided when "
                "creating SpelledMusicBert from an original MusicBERT model."
            )
        if target_vocab_size < original_config.vocab_size:
            raise ValueError(
                "SpelledMusicBert vocab_size cannot be smaller than the original "
                f"MusicBERT vocab size ({original_config.vocab_size})."
            )
        new_config = SpelledMusicBertConfig(
            vocab_size=target_vocab_size,
            compound_ratio=9,
            **{
                k: v
                for k, v in original_config.__dict__.items()
                if k not in ["vocab_size", "compound_ratio"]
            },
        )

        # Create new model
        new_model = cls(new_config, original_model=original_model)

        # Explicitly copy all parameters that aren't in the embeddings
        # This ensures any weights not handled by the initialization logic are preserved
        with torch.no_grad():
            for name, param in original_model.named_parameters():
                if "embeddings" not in name and hasattr(new_model, name):
                    target_param = getattr(new_model, name)
                    if param.shape == target_param.shape:
                        target_param.copy_(param)

        return new_model

    @classmethod
    def create_from_checkpoint(
        cls,
        checkpoint_path: str,
        spelling_tokens: Optional[List[str]] = None,
        vocab_size: Optional[int] = None,
    ) -> Tuple["SpelledMusicBert", List[str]]:
        """Class method to create instance from checkpoint"""
        return create_spelled_musicbert_from_checkpoint(
            checkpoint_path, spelling_tokens, vocab_size=vocab_size
        )


def create_spelled_musicbert_from_checkpoint(
    checkpoint_path: str,
    spelling_tokens: list = None,
    vocab_size: Optional[int] = None,
) -> Tuple[SpelledMusicBert, List[str]]:
    """
    Create a SpelledMusicBert from an original MusicBert checkpoint

    Args:
        checkpoint_path: Path to original MusicBert checkpoint
        spelling_tokens: Accidental tokens; defaults include single, double,
            and triple sharps/flats plus the natural sign.
        vocab_size: Optional concrete event vocabulary size. Use this when
            spelling token ids are not exactly original_vocab_size..N.

    Returns:
        Tuple containing:
            - SpelledMusicBert model ready for training with spelled pitch
            - List of spelling tokens added to the vocabulary
    """
    from musicbert_hf.checkpoints import load_musicbert_from_fairseq_checkpoint

    if spelling_tokens is None:
        spelling_tokens = list(DEFAULT_SPELLING_TOKENS)
    if vocab_size is None:
        vocab_size = event_vocab_size(spelling_tokens)

    logger.info(f"Loading base MusicBert model from {checkpoint_path}")
    logger.info(f"Adding {len(spelling_tokens)} spelling tokens: {spelling_tokens}")

    # Load original model
    original_model = load_musicbert_from_fairseq_checkpoint(
        checkpoint_path, print_missing_keys=True
    )

    # Create spelled version
    spelled_model = SpelledMusicBert.from_original_musicbert(
        original_model,
        spelling_vocab_size=len(spelling_tokens),
        vocab_size=vocab_size,
    )

    return spelled_model, spelling_tokens


def load_spelled_musicbert_checkpoint(
    checkpoint_path: str,
    spelling_tokens: Optional[List[str]] = None,
    vocab_size: Optional[int] = None,
) -> Tuple[SpelledMusicBert, List[str]]:
    """Load a spelled backbone from an HF encoder directory or a base checkpoint."""
    if os.path.isdir(checkpoint_path) and os.path.exists(
        os.path.join(checkpoint_path, "config.json")
    ):
        model = SpelledMusicBert.from_pretrained(checkpoint_path)
        return model, spelling_tokens or list(DEFAULT_SPELLING_TOKENS)
    return create_spelled_musicbert_from_checkpoint(
        checkpoint_path, spelling_tokens, vocab_size=vocab_size
    )


def load_finetuned_musicbert_checkpoint(
    base_checkpoint_path: str,
    checkpoint_directory: str,
    *,
    use_spelled_musicbert: bool,
    spelling_tokens: Optional[List[str]] = None,
    freeze_layers_rnbert="none",
    torch_dtype=torch.bfloat16,
) -> MusicBertBackbone:
    """Restore an encoder using saved tensor sizes and validate before copying.

    Frozen parameters may be supplied by the base checkpoint, matching the
    existing save format, which stores only trainable encoder parameters.
    """
    weights = {}
    if os.path.isdir(checkpoint_directory):
        weights = _load_weights_from_path(checkpoint_directory)
        if not weights:
            raise RuntimeError(f"Could not read encoder weights: {checkpoint_directory}")
    elif freeze_layers_rnbert != "all":
        raise FileNotFoundError(f"Encoder checkpoint directory not found: {checkpoint_directory}")

    # Only the obsolete MLM head is ignored. All backbone parameters retain the
    # same strict validation for missing, unexpected, or mismatched tensors.
    weights = {name: tensor for name, tensor in weights.items()
               if not name.startswith("cls.")}

    embedding_key = "bert.embeddings.word_embeddings.weight"
    embedding_weight = weights.get(embedding_key)
    if embedding_weight is not None and embedding_weight.ndim != 2:
        raise ValueError(f"Expected a matrix for {embedding_key}")
    vocab_size = embedding_weight.shape[0] if embedding_weight is not None else None

    if use_spelled_musicbert:
        model, _ = load_spelled_musicbert_checkpoint(
            base_checkpoint_path, spelling_tokens=spelling_tokens, vocab_size=vocab_size
        )
    else:
        from musicbert_hf.checkpoints import load_musicbert_from_fairseq_checkpoint

        original_model = load_musicbert_from_fairseq_checkpoint(base_checkpoint_path)
        model = MusicBertBackbone(original_model.config, original_model.bert)

    # HF base directories can carry a vocabulary different from the saved
    # fine-tuned embedding. Resize only parameters supplied in full by weights.
    rebuilt_parameters = set()
    embedding = model.get_input_embeddings()
    if vocab_size is not None and embedding.num_embeddings != vocab_size:
        model.set_input_embeddings(nn.Embedding(
            vocab_size, embedding.embedding_dim, padding_idx=embedding.padding_idx,
            device=embedding.weight.device, dtype=embedding.weight.dtype,
        ))
        rebuilt_parameters.add(embedding_key)
    if vocab_size is not None:
        model.config.vocab_size = vocab_size

    model = model.to(torch_dtype)
    if freeze_layers_rnbert == "all":
        for parameter in model.parameters():
            parameter.requires_grad = False
    elif freeze_layers_rnbert is None or freeze_layers_rnbert == "none":
        for parameter in model.parameters():
            parameter.requires_grad = True
    else:
        freeze_layers(model, freeze_layers_rnbert, freeze_cls=True)

    expected = model.state_dict()
    unexpected = sorted(set(weights) - set(expected))
    missing = [name for name, parameter in model.named_parameters()
               if (parameter.requires_grad or name in rebuilt_parameters) and name not in weights]
    mismatched = [name for name in weights.keys() & expected.keys()
                  if weights[name].shape != expected[name].shape]
    if unexpected or missing or mismatched:
        raise ValueError(
            f"Invalid encoder checkpoint {checkpoint_directory}: "
            f"missing trainable parameters={missing}; unexpected keys={unexpected}; "
            f"shape mismatches={mismatched}"
        )
    model.load_state_dict(weights, strict=False)
    logger.info("Restored encoder from %s (embedding vocab=%s).",
                checkpoint_directory, model.get_input_embeddings().num_embeddings)
    return model
