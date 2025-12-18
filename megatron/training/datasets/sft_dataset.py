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
        if conversation_list["stage"] == "1":
            tokens, target = tokenizer.tokenize_conversation_for_stage_1(
                conversation_list["query_and_response"]
            )
            
        elif conversation_list["stage"] == "2":
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
            if True:  # TODO: when too long to fit in context, truncate left to right
                tokens = tokens[: max_seq_len - 1]
                target = target[: max_seq_len - 1]
            else:  # right to left
                tokens = tokens[-(max_seq_len - 1) :]
                target = target[-(max_seq_len - 1) :]

        total_len = len(target) # 마지막 eos까지의 길이

        # padding
        num_tokens = len(tokens) + 1
        padding_len = max_seq_len - num_tokens
        assert padding_len >= 0
        filler = [tokenizer.pad] * (padding_len + 1)

        tokens = tokens.tolist() + [tokenizer.eod] + filler
        target = target.tolist() + [tokenizer.eod] + filler

        tokens = torch.tensor(tokens)
        target = torch.tensor(target)

        tokens = tokens[:-1].contiguous()
        target = target[1:].contiguous()

        # TODO: mask template parts
        loss_mask, position_ids = self._get_ltor_masks_and_position_ids(
            max_seq_len, target, tokenizer.pad, conversation_list['stage'], total_len
        )

        ret = {
            'tokens': tokens,
            'labels': target,
            'loss_mask': loss_mask,
            'position_ids': position_ids,
        }

        return ret

    def _get_ltor_masks_and_position_ids(self, max_seq_len, target, pad_token, stage, total_len):
        """Build masks and position id for left to right model for SFT"""

        assert not self.config.reset_position_ids and not self.config.reset_attention_mask

        # Position ids.
        position_ids = torch.arange(max_seq_len, dtype=torch.long)

        # Loss mask.
        loss_mask = torch.ones(max_seq_len, dtype=torch.float)

        if stage == "1":
            loss_mask[total_len-1:] = 0.0

        elif stage == "2":
            # 첫 번째 EOS는 진짜 문장 끝이므로 loss 계산
            # 그 이후의 PAD(=EOS)는 마스킹
            eos_positions = (target == pad_token).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                first_eos = eos_positions[0].item()
                # 첫 번째 EOS까지는 loss 계산 (loss_mask = 1.0 유지)
                # 첫 번째 EOS 이후는 마스킹
                loss_mask[first_eos + 1:] = 0.0

        # loss_mask[target == IGNORE_INDEX] = 0.0  # mask prompts

        return loss_mask, position_ids
