from copy import deepcopy
from typing import Dict, List, Tuple

import torch
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...util import nethook
from ...util.generate import generate_fast

from .compute_u import compute_u
from .compute_v import compute_v
from .rome_lora_hparams import ROMELoRAHyperParams

CONTEXT_TEMPLATES_CACHE = None



def svd_update(delta_W, r):


    # 进行 SVD
    U, S, V = torch.linalg.svd(delta_W, full_matrices=False)

    # 选择秩 r
    U_r = U[:, :r]
    S_r = torch.diag(torch.sqrt(S[:r]))
    V_r = V[:, :r]

    # 计算 Delta W_A 和 Delta W_B
    delta_W_A = U_r @ S_r
    delta_W_B = S_r @ V_r.T

    print("Shape Delta W_A:", delta_W_A)
    print("Shape Delta W_B:", delta_W_B)

    # 验证 Delta W_A * Delta W_B ≈ Delta W
    reconstructed_delta_W = delta_W_A @ delta_W_B
    print("Reconstructed Delta W:", reconstructed_delta_W)
    
    return delta_W_A, delta_W_B



def get_upd_for_loraA(param_loraB, upd_W):
    
    B_pinv = torch.linalg.pinv(param_loraB) 
    upd_A = upd_W @ B_pinv
    return upd_A

def get_upd_for_loraB(param_loraA, upd_W):
    A_pinv = torch.linalg.pinv(param_loraA)
    upd_B = A_pinv @ upd_W
    return upd_B

def get_upd_for_loraA_and_B(param_loraA, param_loraB, upd_W, hparams):
    
    upd_A = torch.randn(param_loraA.size(), dtype=torch.float32, requires_grad=True)
    upd_B = torch.randn(param_loraB.size(), dtype=torch.float32, requires_grad=True)
    
    optimizer = optim.Adam([upd_A, upd_B], lr=hparams.lora_lr)
    
    
    for iter in range(hparams.lora_num_grad_steps):

        optimizer.zero_grad()
        # Compute the current prediction
        cur_upd_w = param_loraA @ upd_B + upd_A @ param_loraB + upd_A @ upd_B
        # Compute the loss function (Euclidean distance)
        loss = (cur_upd_w - upd_W).pow(2).sum()
        # Backpropagation
        loss.backward()
        # Update delta A and delta B
        optimizer.step()
    
    return upd_A.detach(), upd_B.detach()

def get_upd_from_updW(param_loraA, param_loraB, upd_W, hparams):
    
    lora_A_zeros = torch.zeros_like(param_loraA)
    lora_B_zeros = torch.zeros_like(param_loraB)
    
    
    if hparams.lora_A_or_B == "lora_A":
        return get_upd_for_loraA(param_loraB, upd_W), lora_B_zeros
    elif hparams.lora_A_or_B == "lora_B":
        return lora_A_zeros, get_upd_for_loraB(param_loraA, upd_W)
    elif hparams.lora_A_or_B == "Both":
        return get_upd_for_loraA_and_B(param_loraA, param_loraB, upd_W, hparams)
    else:
        raise ValueError(f"fact_token={hparams.fact_token} not recognized")
    
    
def apply_rome_to_lora_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: List[Dict],
    hparams: ROMELoRAHyperParams,
    copy=False,
    return_orig_weights=False,
    keep_original_weight=False,
    **kwargs
) -> Tuple[AutoModelForCausalLM, List[str]]:
    """
    Returns a model with the desired changes.

    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.

    :return: (1) the updated model, (2) an original copy of the weights that changed
    """
    request = request[0]
    if copy:
        model = deepcopy(model)

    weights_copy = {}

    deltas = execute_rome_lora(model, tok, request, hparams)

    with torch.no_grad():
        for w_name, (delta_u, delta_v) in deltas.items():
            loraA_name, loraB_name = w_name.split(" ")
            upd_matrix = delta_u.unsqueeze(1) @ delta_v.unsqueeze(0)

            params_loraA = nethook.get_parameter(model, loraA_name)
            params_loraB = nethook.get_parameter(model, loraB_name)
            
            upd_loraA, upd_loraB = get_upd_from_updW(param_loraA=params_loraA, param_loraB=params_loraB, upd_W=upd_matrix, hparams=hparams)
            
            upd_loraA = upd_matrix_match_shape(upd_loraA, params_loraA.shape)
            upd_loraB = upd_matrix_match_shape(upd_loraB, params_loraB.shape)

            if return_orig_weights and w_name not in weights_copy:
                raise ValueError("Not Implemented")
                weights_copy[w_name] = w.detach().clone()

            params_loraA[...] += upd_loraA
            params_loraB[...] += upd_loraB
            # w[...] += upd_matrix

        print(f"New weights successfully inserted into {list(deltas.keys())}")

    if not keep_original_weight:
        weights_copy = {}

    return model, weights_copy


def execute_rome_lora(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: ROMELoRAHyperParams,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the ROME update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    # Update target and print info
    request = deepcopy(request)
    if request["target_new"] != " ":
        # Space required for correct tokenization
        request["target_new"] = " " + request["target_new"]

    if '{}' not in request['prompt']:
        assert request['subject'] in request['prompt'] or \
               print(f"Subject:{request['subject']} do not exist in prompt: {request['prompt']}")

        request['prompt'] = request['prompt'].replace(request['subject'], '{}')

    print(
        f"Executing ROMELoRA algorithm for the update: "
        f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']}]"
    )

    # Retrieve weights that user desires to change
    weights = {
        f"{module.format(layer)}.weight": nethook.get_parameter(
            model, f"{module.format(layer)}.weight"
        )
        for module in hparams.target_module_tmp
        for layer in hparams.layers
    }
    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}
    
    # Update loop: sequentially intervene at each specified layer
    deltas = {}
    for layer in sorted(hparams.layers):
        # Compute rank-1 update matrix
        left_vector: torch.Tensor = compute_u(
            model,
            tok,
            request,
            hparams,
            layer,
            get_context_templates(model, tok, hparams.context_template_length_params),
        )
        print("Left vector shape:", left_vector.shape)
        right_vector: torch.Tensor = compute_v(
            model,
            tok,
            request,
            hparams,
            layer,
            left_vector,
            get_context_templates(model, tok, hparams.context_template_length_params),
        )
        print("Right vector shape:", right_vector.shape)

        with torch.no_grad():
            
            
            ### 1. get the upd_matrix

            upd_matrix = left_vector.unsqueeze(1) @ right_vector.unsqueeze(0) # shape [4096, 11008]
            
            cov_layer_name = hparams.cov_module_tmp.format(layer)
            cov_module = nethook.get_module(model, cov_layer_name)
            in_featuers, out_features = cov_module.in_features,  cov_module.out_features
            print("Cov module shape:", in_featuers, out_features)

            breakpoint()
            upd_matrix_match_shape(upd_matrix, torch.Size([out_features, in_featuers]))
            
            ### 2. get the delta_W_A and delta_W_B
            assert hparams.lora_A_or_B in ["lora_A", "lora_B", "Both"], "lora_A_or_B should be lora_A, lora_B or Both"
            
            target_module_names = [ module_tmp.format(layer) for module_tmp in hparams.target_module_tmp] 
            params_loraA = nethook.get_parameter(model, target_module_names[0]) + weight
            params_loraB = nethook.get_parameter(model, target_module_names[1])
            
            upd_loraA, upd_loraB = get_upd_from_updW(param_loraA=params_loraA, param_loraB=params_loraB, upd_W=upd_matrix, hparams=hparams)
            
            breakpoint()
            ####3. update lora parameters
            # Update model weights and record desired changes in `delta` variable
            for weight_name, param in zip(target_module_names, [upd_loraA, upd_loraB]):
                weights[weight_name][...] += param

            breakpoint()

            deltas[" ".join(target_module_names)]  = (
                left_vector.detach(),
                right_vector.detach(),
            )

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}")

    # breakpoint()
    return deltas


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError
    """

    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            "Update matrix computed by ROME does not match original weight shape. "
            "Check for bugs in the code?"
        )


def get_context_templates(model, tok, length_params):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = ["{}"] + [
            x.replace("{", "").replace("}", "") + ". {}"
            for x in sum(
                (
                    generate_fast(
                        model,
                        tok,
                        ["The", "Therefore", "Because", "I", "You"],
                        n_gen_per_prompt=n_gen // 5,
                        max_out_len=length,
                    )
                    for length, n_gen in length_params
                ),
                [],
            )
        ]

        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE
