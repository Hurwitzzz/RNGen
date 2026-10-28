#!/usr/bin/env python3
"""Compare MusicBert and SpelledMusicBert outputs on equivalent inputs.

This is a manual verification script, not an automated regression test.
It checks whether adding spelling tokens preserves the original MusicBert
behavior for standard 8-token inputs.
"""

import argparse
import logging
from typing import Dict, Tuple

import torch
from musicbert_hf.checkpoints import load_musicbert_from_fairseq_checkpoint

from rngen.modeling.encoder import SpelledMusicBert
from rngen.event_vocabulary import DEFAULT_SPELLING_TOKENS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare MusicBert and SpelledMusicBert outputs."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the MusicBert checkpoint.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--seq-length",
        type=int,
        default=128,
        help="Input length for the 8-token representation. Must be divisible by 8.",
    )
    parser.add_argument("--num-tests", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--spelling-tokens",
        type=str,
        nargs="+",
        default=list(DEFAULT_SPELLING_TOKENS),
    )
    return parser.parse_args()


def load_models(
    checkpoint_path: str, spelling_tokens: list, device: str
) -> Tuple[torch.nn.Module, torch.nn.Module]:
    logger.info("Loading MusicBert from %s", checkpoint_path)
    musicbert_model = load_musicbert_from_fairseq_checkpoint(
        checkpoint_path, print_missing_keys=True
    )
    musicbert_model.to(device)
    musicbert_model.eval()

    logger.info("Loading SpelledMusicBert from %s", checkpoint_path)
    spelled_musicbert_model, _ = SpelledMusicBert.create_from_checkpoint(
        checkpoint_path,
        spelling_tokens=spelling_tokens,
    )
    spelled_musicbert_model.to(device)
    spelled_musicbert_model.eval()

    return musicbert_model, spelled_musicbert_model


def generate_test_inputs(
    batch_size: int, seq_length: int, device: str, vocab_size: int = 1237
) -> Dict[str, torch.Tensor]:
    assert seq_length % 8 == 0, "Sequence length must be divisible by 8."

    input_ids = torch.randint(
        low=0, high=vocab_size, size=(batch_size, seq_length), device=device
    )
    attention_mask = torch.ones(
        (batch_size, seq_length // 8), dtype=torch.long, device=device
    )
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def convert_inputs_for_spelled(
    musicbert_inputs: Dict[str, torch.Tensor], device: str
) -> Dict[str, torch.Tensor]:
    batch_size, seq_length = musicbert_inputs["input_ids"].shape
    new_seq_length = seq_length + (seq_length // 8)
    spelled_input_ids = torch.zeros(
        (batch_size, new_seq_length), dtype=torch.long, device=device
    )

    for i in range(seq_length // 8):
        src_start = i * 8
        src_end = src_start + 8
        dst_start = i * 9
        dst_end = dst_start + 8
        spelled_input_ids[:, dst_start:dst_end] = musicbert_inputs["input_ids"][
            :, src_start:src_end
        ]
        spelled_input_ids[:, dst_end] = 1238

    return {
        "input_ids": spelled_input_ids,
        "attention_mask": musicbert_inputs["attention_mask"],
    }


def run_comparison_test(
    musicbert_model: torch.nn.Module,
    spelled_musicbert_model: torch.nn.Module,
    test_inputs: Dict[str, torch.Tensor],
    spelled_inputs: Dict[str, torch.Tensor],
    atol: float,
    rtol: float,
) -> bool:
    with torch.no_grad():
        musicbert_outputs = musicbert_model.bert(
            output_hidden_states=False,
            return_dict=True,
            upsample=False,
            **test_inputs,
        )
        spelled_outputs = spelled_musicbert_model.bert(**spelled_inputs)

    musicbert_hidden_states = musicbert_outputs.last_hidden_state
    spelled_hidden_states = spelled_outputs.last_hidden_state

    if musicbert_hidden_states.shape != spelled_hidden_states.shape:
        logger.error(
            "Shape mismatch: MusicBert %s vs SpelledMusicBert %s",
            musicbert_hidden_states.shape,
            spelled_hidden_states.shape,
        )
        return False

    is_close = torch.allclose(
        musicbert_hidden_states,
        spelled_hidden_states,
        atol=atol,
        rtol=rtol,
    )

    avg_diff = torch.mean(
        torch.abs(musicbert_hidden_states - spelled_hidden_states)
    ).item()
    max_diff = torch.max(
        torch.abs(musicbert_hidden_states - spelled_hidden_states)
    ).item()

    if is_close:
        logger.info(
            "Outputs match. Average difference %.8f, max difference %.8f",
            avg_diff,
            max_diff,
        )
    else:
        logger.error(
            "Outputs differ. Average difference %.8f, max difference %.8f",
            avg_diff,
            max_diff,
        )

    return is_close


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    musicbert_model, spelled_musicbert_model = load_models(
        args.checkpoint, args.spelling_tokens, args.device
    )

    all_tests_passed = True
    for test_idx in range(args.num_tests):
        logger.info("Running test %d/%d", test_idx + 1, args.num_tests)
        test_inputs = generate_test_inputs(
            args.batch_size, args.seq_length, args.device
        )
        spelled_inputs = convert_inputs_for_spelled(test_inputs, args.device)
        passed = run_comparison_test(
            musicbert_model,
            spelled_musicbert_model,
            test_inputs,
            spelled_inputs,
            args.atol,
            args.rtol,
        )
        all_tests_passed = all_tests_passed and passed

    if all_tests_passed:
        logger.info("All comparison tests passed.")
    else:
        logger.error("At least one comparison test failed.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
