from dataclasses import dataclass
from typing import List
import yaml

from ...util.hparams import HyperParams


@dataclass
class ROMELoRAHyperParams(HyperParams):
    # Method
    layers: List[int]
    total_layers: List[int]
    fact_token: str
    v_num_grad_steps: int
    v_lr: float
    v_loss_layer: int
    v_weight_decay: float
    clamp_norm_factor: float
    kl_factor: float
    mom2_adjustment: bool
    context_template_length_params: List[List[int]]

    # Module templates
    # rewrite_module_tmp: list[str]
    # layer_module_tmp: str
    # mlp_module_tmp: str
    # 
    
    lora_A_or_B:str
    lora_lr: float
    lora_num_grad_steps: int
    cov_module_tmp: str
    value_module_tmp: str
    target_module_tmp: str
    layer_module_tmp: str
    
    attn_module_tmp: str
    ln_f_module: str
    lm_head_module: str

    # Statistics
    mom2_dataset: str
    mom2_n_samples: int
    mom2_dtype: str
    alg_name: str
    device: int
    model_name: str
    stats_dir: str

    max_length: int = 40
    model_parallel: bool = False
    fp16: bool = False
    
    #local setting
    ckpt_path: str = ''
    mix_module_tmp: str = None
    rank: int = 0,
    #total layer to edit for edit_diff_layer


    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):

        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)

        assert (config and config['alg_name'] == 'ROMELoRA') or print(f'ROMEHyperParams can not load from {hparams_name_or_path}, '
                                                f'alg_name is {config["alg_name"]} ')
        return cls(**config)
