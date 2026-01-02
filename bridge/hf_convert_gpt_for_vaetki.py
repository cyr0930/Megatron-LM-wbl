import time
import os
import shutil
import json
import argparse
import math
import transformers
from itertools import takewhile
from accelerate import init_empty_weights
from huggingface_hub import split_torch_state_dict_into_shards
from transformers.utils import SAFE_WEIGHTS_NAME, SAFE_WEIGHTS_INDEX_NAME
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.training.arguments import add_megatron_arguments
from megatron.bridge import AutoBridge
from megatron.bridge.training.model_load_save import load_model_config, temporary_distributed_context
from megatron.bridge.training.checkpointing import _load_model_weights_from_checkpoint
from pretrain_gpt_for_wbl import model_provider_with_args
from .vaetki_bridge import VaetkiBridge    # register bridge

DIR_SETTINGS = "settings"
DIR_HF = "hf"


def _create_hf_config(model, tokenizer, save_directory):
    config = {
        "auto_map": {
            "AutoConfig": "configuration_vaetki.VaetkiConfig",
            "AutoModel": "modeling_vaetki.VaetkiModel",
            "AutoModelForCausalLM": "modeling_vaetki.VaetkiForCausalLM"
        },
        "architectures": ["VaetkiForCausalLM"],
        "model_type": "vaetki",
        "hidden_act": "silu",
        "attention_bias": False,
        "attention_dropout": model.config.attention_dropout,
        "initializer_range": model.config.init_method_std,
        "num_hidden_layers": model.config.num_layers,
        "first_k_dense_replace": sum(1 for _ in takewhile(lambda x: x == 0, model.config.moe_layer_freq)),
        "hidden_size": model.config.hidden_size,
        "intermediate_size": model.config.ffn_hidden_size,
        "moe_intermediate_size": model.config.moe_ffn_hidden_size,
        "n_routed_experts": model.config.num_moe_experts,
        "n_shared_experts": 1,
        "num_experts_per_tok": model.config.moe_router_topk,
        "norm_topk_prob": True,
        "routed_scaling_factor": model.config.moe_router_topk_scaling_factor or 1.0,
        "sliding_window": model.config.window_size[0],
        "num_attention_heads": model.config.num_attention_heads,
        "qk_nope_head_dim": model.config.qk_head_dim,
        "qk_rope_head_dim": model.config.qk_pos_emb_head_dim,
        "v_head_dim": model.config.v_head_dim,
        "q_lora_rank": model.config.q_lora_rank,
        "kv_lora_rank": model.config.kv_lora_rank,
        "rms_norm_eps": model.config.layernorm_epsilon,
        "rope_interleave": True,
        # TODO: correct weights according to rope_factor
        # TODO: mscale for yarn
        # "rope_scaling": {
        #     "rope_type": model.config.rope_type,
        #     "factor": model.config.rotary_scaling_factor,
        #     "original_max_position_embeddings": model.config.original_max_position_embeddings,
        # },
        "rope_scaling": None,   # disable yarn for now
        "rope_theta_global": float(model.config.rotary_base_global),
        "rope_theta_local": float(model.config.rotary_base),
        "max_position_embeddings": 131072,
        "tie_word_embeddings": model.share_embeddings_and_output_weights,
        "vocab_size": model.vocab_size,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "dtype": "bfloat16",
        "transformers_version": transformers.__version__,
    }
    with open(f"{save_directory}/config.json", "w") as w:
        json.dump(config, w, indent=2)


def _copy_codes(save_directory):
    shutil.copy("./bridge/configuration_vaetki.py", f"{save_directory}/")
    shutil.copy("./bridge/modeling_vaetki.py", f"{save_directory}/")


def _copy_tokenizer_files(tokenizer_path, save_directory):
    for item in os.listdir(tokenizer_path):
        shutil.copy(f"{tokenizer_path}/{item}", f"{save_directory}/")


def _get_max_shard_size(megatron_model_path):
    size = 0
    for name in os.listdir(megatron_model_path):
        fp = os.path.join(megatron_model_path, name)
        if os.path.isfile(fp) and name.endswith(".distcp"):
            size += os.path.getsize(fp)
    if size >= 5 * 1024 ** 3:
        return "5GB"
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            size = math.ceil(size / 2)
            return f"{size}{unit}"
        size /= 1024


def _save_model_shard_idx(model, save_directory, args):
    state_dict = model.state_dict()
    weights_name = SAFE_WEIGHTS_NAME
    filename_pattern = weights_name.replace(".safetensors", "{suffix}.safetensors")
    max_shard_size = _get_max_shard_size(args.megatron_model_path)
    state_dict_split = split_torch_state_dict_into_shards(
        state_dict, filename_pattern=filename_pattern, max_shard_size=max_shard_size
    )
    if state_dict_split.is_sharded:
        index = {
            "metadata": {"total_parameters": model.num_parameters(), **state_dict_split.metadata},
            "weight_map": state_dict_split.tensor_to_filename,
        }
        save_index_file = os.path.join(save_directory, SAFE_WEIGHTS_INDEX_NAME)
        with open(save_index_file, "w", encoding="utf-8") as f:
            content = json.dumps(index, indent=2, sort_keys=True) + "\n"
            f.write(content)


def preprocess(megatron_model, args):
    setting_path = f"{args.hf_model_path}/{DIR_SETTINGS}"
    os.makedirs(setting_path, exist_ok=True)

    _copy_codes(setting_path)
    _copy_tokenizer_files(args.tokenizer_path, setting_path)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    _create_hf_config(megatron_model, tokenizer, setting_path)

    config = AutoConfig.from_pretrained(setting_path, trust_remote_code=True)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    _save_model_shard_idx(model, setting_path, args)


def megatron_to_hf(model, save_directory):
    setting_path = f"{save_directory}/{DIR_SETTINGS}"
    output_path = f"{save_directory}/{DIR_HF}"
    bridge = AutoBridge.from_hf_pretrained(setting_path, trust_remote_code=True)
    bridge.save_hf_pretrained([model], output_path)
    _copy_codes(output_path)


def _merge_missing(src, dst):
    for k, v in vars(src).items():
        if not hasattr(dst, k):
            setattr(dst, k, v)


def load_megatron_model(args):
    defaults = add_megatron_arguments(argparse.ArgumentParser(allow_abbrev=False)).parse_args([])
    _, mlm_args = load_model_config(args.megatron_model_path)
    _merge_missing(defaults, mlm_args)
    mlm_args.use_cpu_initialization = args.use_cpu_initialization

    # TODO: parallel conversion
    mlm_args.context_parallel_size = 1
    mlm_args.expert_model_parallel_size = 1
    mlm_args.expert_tensor_parallel_size = 1
    mlm_args.pipeline_model_parallel_size = 1
    mlm_args.tensor_model_parallel_size = 1
    mlm_args.sequence_parallel = False
    mlm_args.context_parallel_size = 1
    mlm_args.transformer_pipeline_model_parallel_size = 1

    mlm_args.recompute_granularity = None

    # with torch.device("meta"):
    pre_process = parallel_state.is_pipeline_first_stage()
    post_process = parallel_state.is_pipeline_last_stage()
    model = model_provider_with_args(mlm_args, pre_process=pre_process, post_process=post_process)
    model.model_type = ModelType.encoder_or_decoder
    _load_model_weights_from_checkpoint(args.megatron_model_path, [model])
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-model-path", type=str)
    parser.add_argument("--megatron-model-path", type=str)
    parser.add_argument("--tokenizer-path", type=str)
    parser.add_argument('--no-use-cpu-initialization', action='store_false', dest='use_cpu_initialization')
    args = parser.parse_args()

    start = time.time()
    # TODO: run without torchrun
    backend = "gloo" if args.use_cpu_initialization else "nccl"
    with temporary_distributed_context(backend):
        megatron_model = load_megatron_model(args)
        preprocess(megatron_model, args)
        megatron_to_hf(megatron_model, args.hf_model_path)
    print("Elapsed time:", time.time() - start)
