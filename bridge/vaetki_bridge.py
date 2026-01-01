import itertools
import logging
from typing import Iterable, List

from megatron.core import parallel_state
from megatron.core.utils import unwrap_model
from megatron.core.models.gpt.gpt_model import GPTModel

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import (
    MegatronModelBridge,
    HFPreTrained,
    MegatronModel,
    WeightConversionTask,
    _megatron_local_name_to_global,
)
from megatron.bridge.models.conversion.param_mapping import AutoMapping, GatedMLPMapping
from megatron.bridge.models.conversion.utils import (
    get_module_and_param_from_name,
    persistent_buffers,
)
from megatron.bridge.utils.common_utils import print_rank_0


logger = logging.getLogger(__name__)


def get_mapping_list() -> list:
    param_mappings = {
        # Embed
        "embedding.word_embeddings.weight": "model.embed_tokens.weight",
        # Attention
        "decoder.layers.*.input_layernorm.weight": "model.layers.*.input_layernorm.weight",
        "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
        "decoder.layers.*.self_attention.linear_kv_down_proj.weight": "model.layers.*.self_attn.kv_a_proj_with_mqa.weight",
        "decoder.layers.*.self_attention.linear_kv_up_proj.weight": "model.layers.*.self_attn.kv_b_proj.weight",
        "decoder.layers.*.self_attention.linear_kv_up_proj.layer_norm_weight": "model.layers.*.self_attn.kv_a_layernorm.weight",
        "decoder.layers.*.self_attention.kv_layernorm.weight": "model.layers.*.self_attn.kv_a_layernorm.weight",
        "decoder.layers.*.self_attention.post_attn_layernorm.weight": "model.layers.*.post_attention_layernorm.weight",
        # Dense MLP
        "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.pre_mlp_layernorm.weight",
        "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
        # MoE
        "decoder.layers.*.pre_mlp_layernorm.weight": "model.layers.*.pre_mlp_layernorm.weight",
        "decoder.layers.*.mlp.router.weight": "model.layers.*.mlp.gate.weight",
        "decoder.layers.*.mlp.experts.linear_fc2.weight*": "model.layers.*.mlp.experts.*.down_proj.weight",
        "decoder.layers.*.mlp.shared_experts.linear_fc2.weight": "model.layers.*.mlp.shared_experts.down_proj.weight",
        "decoder.layers.*.post_mlp_layernorm.weight": "model.layers.*.post_mlp_layernorm.weight",
        # LM Head
        "decoder.final_layernorm.weight": "model.norm.weight",
        "output_layer.weight": "lm_head.weight",
        # MLA
        "decoder.layers.*.self_attention.linear_q_down_proj.weight": "model.layers.*.self_attn.q_a_proj.weight",
        "decoder.layers.*.self_attention.linear_q_up_proj.weight": "model.layers.*.self_attn.q_b_proj.weight",
        "decoder.layers.*.self_attention.linear_q_up_proj.layer_norm_weight": "model.layers.*.self_attn.q_a_layernorm.weight",
        # Mcore local spec
        "decoder.layers.*.self_attention.q_layernorm.weight": "model.layers.*.self_attn.q_a_layernorm.weight",
        # For models without MLA
        "decoder.layers.*.self_attention.linear_q_proj.weight": "model.layers.*.self_attn.q_proj.weight",
    }

    mapping_list = []
    # Convert each dictionary entry to AutoMapping(hf_param, megatron_param)
    for megatron_param, hf_param in param_mappings.items():
        mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

    mapping_list.extend(
        [
            GatedMLPMapping(
                megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
                gate="model.layers.*.mlp.gate_proj.weight",
                up="model.layers.*.mlp.up_proj.weight",
            ),
            GatedMLPMapping(
                megatron_param="decoder.layers.*.mlp.experts.linear_fc1.weight*",
                gate="model.layers.*.mlp.experts.*.gate_proj.weight",
                up="model.layers.*.mlp.experts.*.up_proj.weight",
            ),
            GatedMLPMapping(
                megatron_param="decoder.layers.*.mlp.shared_experts.linear_fc1.weight",
                gate="model.layers.*.mlp.shared_experts.gate_proj.weight",
                up="model.layers.*.mlp.shared_experts.up_proj.weight",
            ),
        ]
    )

    return mapping_list


@MegatronModelBridge.register_bridge(source="VaetkiForCausalLM", target=GPTModel)
class VaetkiBridge(MegatronModelBridge):

    def mapping_registry(self) -> MegatronMappingRegistry:
        mapping_list = get_mapping_list()
        return MegatronMappingRegistry(*mapping_list)

    def build_conversion_tasks(
        self, hf_pretrained: HFPreTrained, megatron_model: List[MegatronModel]
    ) -> List[None | WeightConversionTask]:

        # Ensure hf_pretrained has the required state structure
        if not (hasattr(hf_pretrained, "state") and hasattr(hf_pretrained.state, "source")):
            raise ValueError("hf_pretrained.state.source is required for weight ordering")

        hf_keys: Iterable[str] = hf_pretrained.state.source.get_all_keys()

        mapping_registry = self.mapping_registry()
        model_unwrapped = unwrap_model(megatron_model)[0]
        model_config = model_unwrapped.config
        embeddings_are_tied = model_config.share_embeddings_and_output_weights = model_unwrapped.share_embeddings_and_output_weights
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        sorted_global_param_names_all_pp_ranks = self._megatron_global_param_names_all_pp_ranks(megatron_model)

        # Filter out output_layer related parameters if embeddings are tied
        if embeddings_are_tied:
            sorted_global_param_names_all_pp_ranks = [
                name for name in sorted_global_param_names_all_pp_ranks if "output_layer" not in name
            ]

        global_names_index_dict = {name: idx for idx, name in enumerate(sorted_global_param_names_all_pp_ranks)}

        tasks = [None] * len(sorted_global_param_names_all_pp_ranks)
        for vp_stage, model in enumerate(megatron_model):
            # persistent buffers are part of the model's state_dict, but not the named_parameters, so we must include them here separately
            for local_name, _ in itertools.chain(model.named_parameters(), persistent_buffers(model)):
                if "_extra_state" in local_name:
                    continue

                local_name = self._unwrap_name(local_name)
                global_name = _megatron_local_name_to_global(megatron_model, model_config, local_name, vp_stage)
                # if name removed due to some reason, continue. e.g. embeddings_are_tied
                if global_name not in global_names_index_dict:
                    print_rank_0(f"WARNING: {global_name} not in global_names_index_dict")
                    continue
                global_name_idx = global_names_index_dict[global_name]
                mapping = mapping_registry.megatron_to_hf_lookup(global_name)

                if not mapping:
                    logger.warning(f"WARNING: No megatron to hf mapping found for {global_name}")
                    continue

                # ensure hf weights exist
                if isinstance(mapping.hf_param, str):
                    if mapping.hf_param not in hf_keys:
                        logger.warning(f"WARNING: Can't find {mapping.hf_param} in hf_keys")
                        continue
                else:
                    missing_params = [hf_param for hf_param in mapping.hf_param.values() if hf_param not in hf_keys]
                    if missing_params:
                        logger.warning(f"WARNING: Can't find the following HF parameters in hf_keys: {missing_params}")
                        continue

                local_module, local_weights = get_module_and_param_from_name(megatron_model, local_name, vp_stage)

                tasks[global_name_idx] = WeightConversionTask(
                    pp_rank=pp_rank,
                    vp_stage=vp_stage,
                    param_name=local_name,
                    megatron_module=local_module,
                    param_weight=local_weights,
                    mapping=mapping,
                )

        # Fill the remaining ones for pp communications
        for idx, global_name in enumerate(sorted_global_param_names_all_pp_ranks):
            mapping = mapping_registry.megatron_to_hf_lookup(global_name)
            if tasks[idx] is None:
                # This is an exception here we pass in global name
                # we are not using global_name to extract module and weights
                # only use it for param mapping auto dispatch checks
                tasks[idx] = WeightConversionTask(
                    pp_rank=pp_rank,
                    vp_stage=None,
                    param_name=global_name,
                    megatron_module=None,
                    param_weight=None,
                    mapping=mapping,
                )

        return tasks
