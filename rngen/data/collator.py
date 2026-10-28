import torch


class DataCollatorForRNGen:
    """Batch and pad RNGen features for training or generation."""

    def __init__(self, tokenizer, llm_pad_token_id, musicbert_pad_token_id):
        self.tokenizer = tokenizer
        self.llm_pad_token_id = llm_pad_token_id
        self.musicbert_pad_token_id = musicbert_pad_token_id

    def __call__(self, features):
        music_input_ids = torch.nn.utils.rnn.pad_sequence(
            [f["music_input_ids"].clone().detach() for f in features],
            batch_first=True,
            padding_value=self.musicbert_pad_token_id,
        )

        music_attention_mask = torch.nn.utils.rnn.pad_sequence(
            [f["music_attention_mask"].clone().detach() for f in features],
            batch_first=True,
            padding_value=0,
        )

        prompt_input_ids = torch.nn.utils.rnn.pad_sequence(
            [f["prompt_input_ids"].clone().detach() for f in features],
            batch_first=True,
            padding_value=self.llm_pad_token_id,
        )

        prompt_attention_mask = torch.nn.utils.rnn.pad_sequence(
            [f["prompt_attention_mask"].clone().detach() for f in features],
            batch_first=True,
            padding_value=0,
        )

        labels = torch.nn.utils.rnn.pad_sequence(
            [f["labels"].clone().detach() for f in features],
            batch_first=True,
            padding_value=-100,
        )

        result = {
            "music_input_ids": music_input_ids,
            "music_attention_mask": music_attention_mask,
            "prompt_input_ids": prompt_input_ids,
            "prompt_attention_mask": prompt_attention_mask,
            "labels": labels,
        }

        if features[0].get("eval_input_ids") is not None:
            eval_input_ids = torch.nn.utils.rnn.pad_sequence(
                [f["eval_input_ids"].clone().detach() for f in features],
                batch_first=True,
                padding_value=self.llm_pad_token_id,
            )

            eval_attention_mask = torch.nn.utils.rnn.pad_sequence(
                [f["eval_attention_mask"].clone().detach() for f in features],
                batch_first=True,
                padding_value=0,
            )

            batch_metadata = [f["sample_metadata"] for f in features]

            result.update(
                {
                    "eval_input_ids": eval_input_ids,
                    "eval_attention_mask": eval_attention_mask,
                    "batch_metadata": batch_metadata,
                }
            )

        return result
