# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

from typing import Any, Dict, Optional

import numpy as np
import torch

from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.megatron_dataset import LowLevelDataset, MegatronDataset
from megatron.core.datasets.utils import Split

IGNORE_INDEX = -100


class SFTLowLevelDataset:
    """The low-level dataset loading jsonl data for SFT

    Args:
        dataset_path (str): The path to jsonl data
            Each line of the jsonl must have key "messages" (List[Dict]),
            which is a sequence of system/user/assistant messages.
            Must be in the following format:
            [
                {"role": "system", "content": "something"},
                {"role": "user", "content": "something1"},
                {"role": "assistant", "content": "something2"},
            ]
    """

    def __init__(self, dataset_path: str = None, dataset=None) -> None:
        try:
            from datasets import load_from_disk
        except ImportError:
            raise ImportError(
                "SFTDataset currently requires datasets library to be installed"
            )
        
        if dataset is not None:
            self.dataset = dataset
        elif dataset_path is not None:
            if dataset_path.endswith(".jsonl"):
                dataset_path = self._jsonl_to_arrow(dataset_path)
            self.dataset = load_from_disk(dataset_path)
        else:
            raise ValueError("Either dataset_path or dataset must be provided")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> list:
        import ast
        
        item = self.dataset[idx]
        
        if 'tools' in item:
            if item['stage'] == "1":
                pass
            elif item['stage'] == "2":
                tools_value = item['tools']
                if isinstance(tools_value, str) and tools_value != "":
                    item['tools'] = ast.literal_eval(tools_value)
        
        if 'query_and_response' in item:
            qr_value = item['query_and_response']
            if item['stage'] == "1":
                pass
            elif item['stage'] == "2":
                if isinstance(qr_value, str) and qr_value != "":
                    item['query_and_response'] = ast.literal_eval(qr_value)
        
        return item
    
    def _jsonl_to_arrow(self, dataset_path):
        import os
        import time
        output_path = dataset_path.replace(".jsonl", "_arrow")
        if os.path.exists(output_path):
            return output_path
        
        global_rank = int(os.getenv("RANK"))
        if global_rank != 0:
            while True:
                time.sleep(5)
                if os.path.exists(output_path):
                    return output_path

        import shutil
        from datasets import load_dataset
        print("Read dataset from jsonl:", dataset_path)
        dataset = load_dataset("json", data_files=dataset_path, split="train")

        output_path_tmp = output_path + "_tmp"
        dataset.save_to_disk(output_path_tmp)
        shutil.move(output_path_tmp, output_path)
        return output_path


class SFTDataset(MegatronDataset):
    """The dataset used during SFT"""

    def __init__(
        self,
        dataset: LowLevelDataset,
        dataset_path: Optional[str],
        indices: np.ndarray,
        num_samples: Optional[int],
        index_split: Split,
        config: GPTDatasetConfig,
    ) -> None:
        super().__init__(dataset, dataset_path, indices, num_samples, index_split, config)

    @staticmethod
    def numel_low_level_dataset(low_level_dataset: LowLevelDataset) -> int:
        return len(low_level_dataset)

    @staticmethod
    def build_low_level_dataset(dataset_path: str, config: GPTDatasetConfig) -> LowLevelDataset:
        return SFTLowLevelDataset(dataset_path)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:

        tokenizer = self.config.tokenizer
        max_seq_len = self.config.sequence_length

        conversation_list = self.dataset[int(self.indices[idx % len(self.indices)])]
        stage = conversation_list['stage']
        if stage == "1":
            tokens, target = tokenizer.tokenize_conversation_for_stage_1(
                conversation_list["query_and_response"]
            )
            
        elif stage == "2":
            if conversation_list["도메인_대분류"] == "8":
                tokens, target = tokenizer.tokenize_conversation(
                    conversation_list["query_and_response"], return_target=True, add_generation_prompt=False, tools=conversation_list["tools"]
                )
            else:
                tokens, target = tokenizer.tokenize_conversation(
                    conversation_list["query_and_response"], return_target=True, add_generation_prompt=False
                )

        # minus one to insert eos token
        if len(tokens) > max_seq_len - 1:
            # if True:  # TODO: when too long to fit in context, truncate left to right
            #     tokens = tokens[: max_seq_len - 1]
            #     target = target[: max_seq_len - 1]
            # else:  # right to left
            tokens = tokens[-(max_seq_len - 1) :]
            target = target[-(max_seq_len - 1) :]
        tokens = tokens.tolist()
        target = target.tolist()

        start_id = tokenizer._tokenizer.vocab["<|START|>"]
        unk_id = tokenizer._tokenizer.vocab["<unk>"]
        role_start_id = tokenizer._tokenizer.vocab["<|role_start|>"]
        role_end_id = tokenizer._tokenizer.vocab["<|role_end|>"]
        think_start_id = tokenizer._tokenizer.vocab["<think>"]
        think_end_id = tokenizer._tokenizer.vocab["</think>"]
        nl_id = tokenizer._tokenizer.vocab["\n"]
        nlnl_id = tokenizer._tokenizer.vocab["\n\n"]
        system_id = tokenizer._tokenizer.encode("system")[0]
        user_id = tokenizer._tokenizer.encode("user")[0]
        assistant_id = tokenizer._tokenizer.encode("assistant")[0]

        loss_mask = []
        role_end, nl_check, keep_mask = True, False, False
        for token_id in target:
            if token_id in [unk_id, start_id, IGNORE_INDEX]:
                loss_mask.append(0.0)
            elif token_id == role_start_id:
                loss_mask.append(0.0)
                role_end = False
            elif token_id == role_end_id:
                loss_mask.append(0.0)
                role_end = True
                nl_check = True
            elif token_id == think_start_id:
                loss_mask.append(0.0)
                nl_check = True
            elif token_id == think_end_id:
                loss_mask.append(1.0)
                nl_check = True
            elif nl_check and token_id in [nl_id, nlnl_id]:
                loss_mask.append(0.0)
                nl_check = False
            elif not role_end:
                loss_mask.append(0.0)
                if token_id in [system_id, user_id]:
                    keep_mask = True
                elif token_id == assistant_id:
                    keep_mask = False
            elif keep_mask:
                loss_mask.append(0.0)
            else:
                loss_mask.append(1.0)

        # padding
        num_tokens = len(tokens) + 1
        padding_len = max_seq_len - num_tokens
        assert padding_len >= 0
        tokens.append(tokenizer.eod)
        target.append(tokenizer.eod)
        loss_mask.append(0.0)
        filler = [tokenizer.pad] * (padding_len + 1)
        tokens.extend(filler)
        target.extend(filler)
        loss_mask.extend([0.0] * len(filler))
        tokens = tokens[:-1]
        target = target[1:]
        loss_mask = loss_mask[1:]

        ret = {
            'tokens': torch.tensor(tokens).contiguous(),
            'labels': torch.tensor(target).contiguous(),
            'loss_mask': torch.tensor(loss_mask),
            'position_ids': torch.arange(max_seq_len, dtype=torch.long),
        }
        return ret
