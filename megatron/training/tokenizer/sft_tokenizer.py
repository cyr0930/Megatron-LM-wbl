# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

"""SFT tokenizer."""
from typing import Dict, List, Union
import numpy as np

nemotron_h_aligned_custom_template = """{% for message in messages %}{% if message['role'] == 'system' %}{{ '<SPECIAL_10>System\n' + message['content'].strip() + '\n' }}{% elif message['role'] == 'user' %}{{ '<SPECIAL_11>User\n' + message['content'].strip() + '\n' + '<SPECIAL_11>Assistant\n' }}{% elif message['role'] == 'assistant' %}{{ message['content'].strip() + '\n' }}{% endif %}{% endfor %}"""

from megatron.core.datasets.megatron_tokenizer import MegatronTokenizer
from megatron.training.datasets.sft_dataset import IGNORE_INDEX
from megatron.training.tokenizer.multimodal_tokenizer import PromptConfig

def load_template(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class SFTTokenizer(MegatronTokenizer):  
    """SFT Tokenizer."""

    def __init__(
        self,
        tokenizer_path: str,
        prompt_format: str,
    ):
        """
        Note: Currently, only HuggingFaceTokenizer is supported as the underlying text tokenizer.

        Args:
            tokenizer_path (str): Underlying tokenizer path.
            prompt_format (str): Prompt format for the tokenizer.
        """
        super().__init__(tokenizer_path, prompt_format=prompt_format)
        try:
            import transformers
        except ImportError:
            raise ImportError(
                "SFTTokenizer currently requires transformers library to be installed"
            )

        # Currently, only HuggingFace tokenizers are supported.
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=tokenizer_path,
        )

        self._vocab_size = len(tokenizer)
        self._tokenizer = tokenizer

        if prompt_format == "nemotron-h-aligned":
            self._prompt_config = PromptConfig(
                assistant_prefix_len=0,
                pad_token_id=tokenizer.convert_tokens_to_ids("<SPECIAL_233>"),
                custom_chat_template=nemotron_h_aligned_custom_template,
                has_bos=False,
                has_system_role=True,
            )
        elif prompt_format == "wbl":
            self._prompt_config = PromptConfig(
                assistant_prefix_len=6,
                pad_token_id=tokenizer.convert_tokens_to_ids("<|END|>"),
                custom_chat_template=load_template(f"{tokenizer_path}/chat_template.jinja"),
                has_bos=False,
                has_system_role=True,
            )
        else:
            raise NotImplementedError("unknown SFT prompt format", prompt_format)

        self._prompt_format = prompt_format

    def tokenize_conversation_for_stage_1(
        self, serialized_conversations
    ):
        whole_tokens = self._tokenizer(
            serialized_conversations,
            return_tensors="np",
            return_token_type_ids=False,
            return_attention_mask=False,
        )["input_ids"][0]
        return whole_tokens, whole_tokens


    def tokenize_conversation(
        self, conversation: List[Dict], return_target: bool, add_generation_prompt: bool, tools=None
    ):
        """Convert a conversation to tokens.

        Args:
            conversation (List[Dict]): Sequence of system/user/assistant messages.
                Must be in the following format:
                [
                    {"role": "system", "content": "something"},
                    {"role": "user", "content": "something1"},
                    {"role": "assistant", "content": "something2"},
                ]
            return_target (bool): Return target tokens with system and assistant masked.
            add_generation_prompt (bool): Add assistant prefix to the end.
        """
        # Skip system message if the tokenizer doesn't have a system role.
        if not self._prompt_config.has_system_role and conversation[0]["role"] == "system":
            conversation = conversation[1:]

        whole_tokens = np.array([], dtype=np.int64)
        buf = []
        for utter in conversation:
            buf.append(utter)
            if utter["role"] == "assistant":
                tokens = self._tokenizer.apply_chat_template(
                    buf,
                    tokenize=True,
                    add_generation_prompt=False,
                    return_assistant_token_mask=False,
                    return_tensors="np",
                    chat_template=self._prompt_config.custom_chat_template,
                    tools=tools
                )[0]
                whole_tokens = np.concatenate((whole_tokens, tokens), axis=0)
                buf = []

        return whole_tokens, whole_tokens

    def tokenize(self, text: Union[str, List[Dict]]):
        """Tokenize conversation or string input."""
        if isinstance(text, list):
            # This code path is used by the inference code currently.
            return self.tokenize_conversation(text, return_target=False, add_generation_prompt=True).tolist()

        return self._encode(text)

    def _encode(self, text: str):
        """Tokenize text input, w/o chat template"""
        return self._tokenizer.encode(text)

    def convert_tokens_to_ids(self, tokens: List[str]):
        """Convert tokens to IDs."""
        return self._tokenizer.convert_tokens_to_ids(tokens)

    def detokenize(self, tokens: List[int]):
        """Detokenize tokens."""
        return self._tokenizer.decode(tokens)

    def get_special_tokens(self):
        """Get special tokens."""
        return self._tokenizer.get_added_vocab()

    @property
    def pad(self):
        """Pad token ID."""
        return self._prompt_config.pad_token_id

    @property
    def eod(self):
        """End of sentence token ID."""
        return self._tokenizer.eos_token_id

    @property
    def vocab(self):
        """Vocab."""
        return NotImplementedError("not used")

    @property
    def inv_vocab(self):
        """Inverse vocab."""
        return NotImplementedError("not used")

    @property
    def vocab_size(self):
        """Vocabulary size."""
        return self._vocab_size
