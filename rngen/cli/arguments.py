from dataclasses import dataclass, field
from typing import List, Optional

TASK_CHOICES = (
    "key_sft",
    "rna_sft",
    "rna_curriculum_stage2_sft",
)
TASK_CHOICES_HELP = ", ".join(f"'{task}'" for task in TASK_CHOICES)
GENERATION_ARG_NAMES = (
    "do_sample",
    "temperature",
    "top_p",
    "num_beams",
    "repetition_penalty",
    "max_new_tokens",
)


def validate_task_name(task: str) -> str:
    """Validate a configured task name against the supported task registry."""
    if task not in TASK_CHOICES:
        raise ValueError(
            f"Unsupported task '{task}'. Expected one of: {', '.join(TASK_CHOICES)}"
        )
    return task


def build_generation_kwargs(source) -> dict:
    """Extract generation kwargs from an object with generation-style attributes."""
    kwargs = {}
    for name in GENERATION_ARG_NAMES:
        if hasattr(source, name):
            value = getattr(source, name)
            if value is not None:
                kwargs[name] = value
    return kwargs


@dataclass
class ModelArguments:
    """Arguments related to model configuration."""

    rnbert_checkpoint_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the RNBert checkpoint"}
    )
    load_from_fted_rnbert: bool = field(
        default=False,
        metadata={
            "help": (
                "Run-name label only: indicates a fine-tuned RNBert source but does "
                "not select or load weights. rnbert_checkpoint_path selects the "
                "encoder when creating a model; finetuned_rngen_checkpoint "
                "loads an existing RNGen model."
            )
        },
    )
    use_spelled_musicbert: bool = field(
        default=False,
        metadata={
            "help": "Whether to use SpelledMusicBert (9-token encoding with accidental tokens) instead of standard MusicBert (8-token encoding)"
        },
    )
    spelling_tokens: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Spelling tokens to add when using SpelledMusicBert"},
    )
    spelling_downsampling_scale: float = field(
        default=1.0,
        metadata={
            "help": "Scale applied to the ninth Octuple+ downsampling slice after loading a SpelledMusicBert checkpoint. Use 1.0 to keep it, 0.0 to zero it, or an intermediate value for scaled transfer."
        },
    )
    include_spelling_in_prompt: bool = field(
        default=False,
        metadata={
            "help": "Whether to include spelling tokens in the prompt for RNGen"
        },
    )
    finetuned_rngen_checkpoint: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to an RNGen checkpoint for resumed training or evaluation. Optimizer and scheduler state are not restored."
        },
    )
    llm_model_name: str = field(
        default="Qwen/Qwen2.5-7B", metadata={"help": "Name of the LLM model to use"}
    )
    projector_layers: int = field(
        default=2, metadata={"help": "Number of layers in the projector"}
    )
    projector_dropout: float = field(
        default=0.1, metadata={"help": "Dropout probability in the projector"}
    )
    freeze_layers_rnbert: str = field(
        default="9",
        metadata={
            "help": "Layers to freeze in RNBert model (int for first N layers, list of ints [1,2,3] for specific layers, 'all' to freeze all layers, 'none' to freeze no layers)"
        },
    )
    freeze_llm: bool = field(
        default=True, metadata={"help": "Whether to freeze LLM parameters"}
    )
    freeze_projector: bool = field(
        default=False, metadata={"help": "Whether to freeze projector parameters"}
    )
    use_lora: bool = field(
        default=False, metadata={"help": "Whether to use LoRA for LLM fine-tuning"}
    )
    lora_r: int = field(default=8, metadata={"help": "LoRA rank parameter"})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA alpha parameter"})
    lora_dropout: float = field(
        default=0.1, metadata={"help": "Dropout probability for LoRA layers"}
    )
    lora_target_modules: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of module names to apply LoRA to (if None, will use defaults)"
        },
    )
    # logs
    wandb_project: Optional[str] = field(
        default=None, metadata={"help": "WandB project name for logging"}
    )
    early_stopping_patience: Optional[int] = field(
        default=None,
        metadata={"help": "If set, add an EarlyStoppingCallback with this patience."},
    )


@dataclass
class DataArguments:
    """Arguments related to data configuration."""

    task: str = field(
        metadata={
            "help": f"The task to perform. Possible values: {TASK_CHOICES_HELP}."
        },
    )
    jsonl_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Stage directory containing train.json, valid.json, and test.json annotations."},
    )
    max_text_length: int = field(
        default=512, metadata={"help": "Maximum length of text tokens"}
    )
    prompt_template: Optional[str] = field(
        default='Group the notes within one bar by their simultaneous onset. Output JSON in the format: [{"position": "<onset_time>", "notes": ["<note1>", "<note2>", ...]}]. <music></music> Answer:',
        metadata={"help": "Template for the prompt with <music> placeholders"},
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Limit the number of samples in the evaluation dataset"},
    )


@dataclass
class GenerationArguments:
    """Arguments related to generation/inference."""

    do_sample: bool = field(
        default=False,
        metadata={
            "help": "Whether to use sampling for generation instead of greedy decoding."
        },
    )

    temperature: float = field(
        default=1.0,
        metadata={"help": "The value used to module the next token probabilities."},
    )

    top_p: float = field(
        default=1.0,
        metadata={
            "help": "The cumulative probability of parameter highest probability vocabulary tokens to keep for nucleus sampling."
        },
    )

    num_beams: int = field(
        default=1,
        metadata={"help": "Number of beams for beam search. 1 means no beam search."},
    )

    repetition_penalty: float = field(
        default=1.0,
        metadata={
            "help": "The parameter for repetition penalty. 1.0 means no penalty."
        },
    )

    max_new_tokens: Optional[int] = field(
        default=512, metadata={"help": "Maximum new tokens for generation"}
    )
