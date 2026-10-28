import json
import os
from typing import List

import torch
from torch.utils.data import Dataset
from rngen.event_vocabulary import load_event_vocab


class RNGenSFTDataset(Dataset):
    """SFT dataset using Qwen templates and on-the-fly event tokenization."""

    def __init__(
        self,
        json_path: str,
        tokenizer,
        compound_ratio: int,
        data_args,
        max_text_length: int = 512,
        is_eval: bool = False,
        include_spelling_in_prompt: bool = False,
    ):
        self.tokenizer = tokenizer
        self.compound_ratio = compound_ratio
        self.max_music_length = max_text_length * self.compound_ratio
        self.max_text_length = max_text_length
        self.is_eval = is_eval
        self.include_spelling_in_prompt = include_spelling_in_prompt
        self.json_path = json_path
        self.task = data_args.task

        with open(json_path, "r") as f:
            self.samples = json.load(f)

        self.template = self.samples["template"]
        self.num_samples = (
            len(self.samples) - 2
            if self.task == "rna_curriculum_stage2_sft"
            else len(self.samples) - 1
        )
        self._initialize_event_tokenizer()

    def _initialize_event_tokenizer(self):
        vocab_path = os.path.join(
            os.path.dirname(__file__),
            "events_incl_spelling_stoi.json",
        )
        if os.path.exists(vocab_path):
            self.event_vocab = load_event_vocab()
        else:
            raise NotImplementedError(
                f"Vocabulary file not found at {vocab_path}. We haven't support generating vocabulary on-the-fly. Please provide a valid vocabulary."
            )

    def _tokenize_events(self, events_str: str) -> List[int]:
        if not events_str:
            return []

        tokens = events_str.split()
        token_ids = []

        for token in tokens:
            if token in self.event_vocab:
                token_ids.append(self.event_vocab[token])
            else:
                raise ValueError(
                    f"Token '{token}' not found in vocabulary. Please check your events data."
                )

        return token_ids

    def __len__(self):
        return self.num_samples

    def _format_curriculum_instruction(self, sample):
        task_name = sample.get("task_name")
        task_templates = self.template.get("tasks", {})
        if task_name not in task_templates:
            raise ValueError(
                f"Unknown RNA curriculum Stage 2 task '{task_name}'. "
                f"Expected one of: {', '.join(sorted(task_templates))}"
            )

        task_template = task_templates[task_name]
        reference_input = sample.get("reference_input", [])
        instruction_template = task_template["instruction"]
        reference_json = json.dumps(
            reference_input,
            indent=None,
            separators=(",", ":"),
            ensure_ascii=False,
        )

        placeholder_count = instruction_template.count("{}")
        if placeholder_count == 1:
            return instruction_template.replace("{}", sample["instruction"], 1)
        if placeholder_count == 2:
            instruction = instruction_template.replace("{}", sample["instruction"], 1)
            return instruction.replace("{}", reference_json, 1)
        raise ValueError(
            f"Unexpected number of placeholders ({placeholder_count}) in "
            f"curriculum instruction for task '{task_name}'."
        )

    def __getitem_curriculum_stage2__(self, idx):
        sample = self.samples[str(idx)].copy()
        if "seq_id" in sample:
            seq_data = self.samples["seq_data"][str(sample["seq_id"])]
            events_str = seq_data["events"]
            spelling = seq_data.get("spelling", "")
            note_count = seq_data.get("note_count", 0)
        else:
            seq_data = {}
            events_str = sample["events"]
            spelling = sample.get("spelling", "")
            note_count = sample.get("note_count", 0)

        music_token_ids = self._tokenize_events(events_str)
        if len(music_token_ids) > self.max_music_length:
            music_token_ids = music_token_ids[: self.max_music_length]

        music_input_ids = torch.tensor(music_token_ids)

        seq_len = len(music_token_ids)
        if seq_len % self.compound_ratio != 0:
            raise ValueError(
                f"Token sequence length ({seq_len}) must be a multiple of {self.compound_ratio}"
            )
        mask_len = seq_len // self.compound_ratio
        music_attention_mask = torch.ones(mask_len, dtype=torch.long)

        instruction = self._format_curriculum_instruction(sample)

        query = [instruction, self.template["input"]]
        if self.include_spelling_in_prompt and spelling:
            query.append(self.template["spelling"].format(spelling))

        user_content = "\n".join(query)
        assistant_content = json.dumps(
            sample["output"],
            indent=None,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        system_content = self.template["system"]

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]

        full_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
            return_tensors="pt",
        )[0]
        full_attention_mask = torch.ones_like(full_ids)

        prompt_messages = messages[:-1]
        prompt_ids = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
        )[0]

        labels = full_ids.clone()
        labels[: len(prompt_ids)] = -100

        result = {
            "music_input_ids": music_input_ids,
            "music_attention_mask": music_attention_mask,
            "prompt_input_ids": full_ids,
            "prompt_attention_mask": full_attention_mask,
            "labels": labels,
        }

        if self.is_eval:
            metadata = sample.get("metadata", {})
            result.update(
                {
                    "eval_input_ids": prompt_ids,
                    "eval_attention_mask": torch.ones_like(prompt_ids),
                    "sample_metadata": {
                        "events": events_str,
                        "note_count": note_count,
                        "spelling": spelling,
                        "reference_output": sample["output"],
                        "reference_input": sample.get("reference_input", []),
                        "task_name": sample.get("task_name"),
                        "sample_id": str(idx),
                        "seq_id": str(sample.get("seq_id", "")),
                        "source_sample_id": metadata.get("source_sample_id"),
                        "source_split": metadata.get("source_split"),
                        "seq_metadata": seq_data.get("metadata", {}),
                        "metadata": metadata,
                    },
                }
            )

        return result

    def __getitem__(self, idx):
        if self.task == "rna_curriculum_stage2_sft":
            return self.__getitem_curriculum_stage2__(idx)

        sample = self.samples[str(idx)].copy()
        events_str = sample["events"]

        music_token_ids = self._tokenize_events(events_str)
        if len(music_token_ids) > self.max_music_length:
            music_token_ids = music_token_ids[: self.max_music_length]

        music_input_ids = torch.tensor(music_token_ids)

        seq_len = len(music_token_ids)
        assert (
            seq_len % self.compound_ratio == 0
        ), f"Check you data please. Token sequences length should be multiple of {self.compound_ratio} for compound mode"
        mask_len = seq_len // self.compound_ratio
        music_attention_mask = torch.ones(mask_len, dtype=torch.long)

        sample["instruction"] = self.template["instruction"].format(
            sample["instruction"]
        )

        query = [sample["instruction"], self.template["input"]]
        if self.include_spelling_in_prompt and sample["spelling"]:
            query.append(self.template["spelling"].format(sample["spelling"]))

        user_content = "\n".join(query)

        if isinstance(sample["output"], dict):
            assistant_content = json.dumps(
                sample["output"], indent=None, separators=(",", ":")
            )
        else:
            assistant_content = sample["output"]

        system_content = self.template["system"]

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]

        full_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
            return_tensors="pt",
        )[0]
        full_attention_mask = torch.ones_like(full_ids)

        prompt_messages = messages[:-1]
        prompt_ids = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
        )[0]

        labels = full_ids.clone()
        labels[: len(prompt_ids)] = -100

        result = {
            "music_input_ids": music_input_ids,
            "music_attention_mask": music_attention_mask,
            "prompt_input_ids": full_ids,
            "prompt_attention_mask": full_attention_mask,
            "labels": labels,
        }

        if self.is_eval:
            eval_input_ids = prompt_ids
            eval_attention_mask = torch.ones_like(eval_input_ids)

            result.update(
                {
                    "eval_input_ids": eval_input_ids,
                    "eval_attention_mask": eval_attention_mask,
                    "sample_metadata": {
                        "events": events_str,
                        "note_count": sample.get("note_count", 0),
                        "spelling": sample.get("spelling", ""),
                        "target_bars": sample.get("target_bars", []),
                        "reference_output": sample["output"],
                        "task_name": sample.get(
                            "task_name", sample.get("metadata", {}).get("task_name")
                        ),
                        "metadata": sample.get("metadata", {}),
                        "sample_id": str(idx),
                        "last_target_bar": sample["metadata"].get(
                            "last_target_bar", None
                        ),
                    },
                }
            )

        return result

    def get_sample_events(self, idx: int) -> str:
        return self.samples[str(idx)]["events"]

    def get_sample_metadata(self, idx: int) -> dict:
        sample = self.samples[str(idx)]
        return {
            "events": sample["events"],
            "note_count": sample.get("note_count", 0),
            "metadata": sample.get("metadata", {}),
            "spelling": sample.get("spelling", ""),
            "target_bars": sample.get("target_bars", []),
            "reference_output": sample["output"],
            "task_name": sample.get(
                "task_name", sample.get("metadata", {}).get("task_name")
            ),
        }
