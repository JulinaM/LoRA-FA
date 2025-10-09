from peft import get_peft_model, LoraConfig, AdaLoraConfig, TaskType
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from utils import (
    train_text_to_text_model,
    model_inference,
    initialize_text_to_text_model,
    transform_dataset,
    merge_llama,
    merge_t5,
)
import json
import math
from datasets import load_dataset
import wandb
from data import *
from typing import List
import torch
import torch.nn as nn
from copy import deepcopy
import logging
from tqdm import tqdm, trange
from typing import Tuple, List, Dict
from peft.tuners.lora.layer import Linear as LoraLinear
from contextlib import contextmanager
from accelerate import Accelerator
import torch.nn.functional as F

log = logging.getLogger(__name__)

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def find_all_linear_modules(model) -> List[str]:
    r"""
    Finds all available modules to apply lora.
    """
    linear_cls = torch.nn.Linear

    output_layer_names = ["lm_head", "embed_tokens"]

    module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, linear_cls) and not any(
            [output_layer in name for output_layer in output_layer_names]
        ):
            module_names.add(name.split(".")[-1])
    return list(module_names)

def get_llama_last_layers(model):
    """
    Returns:
        last_block_linear: list of nn.Linear names in last transformer block
    """
    last_block = model.model.layers[-1]
    last_block_linear = []
    for name, module in last_block.named_modules():
        if isinstance(module, torch.nn.Linear):
            full_name = f"model.layers.{len(model.model.layers)-1}.{name}"
            last_block_linear.append(full_name)
    return last_block_linear

def remove_dropout(module):
    print('-> removing droupouts ')
    for name, child in module.named_children():
        if isinstance(child, nn.Dropout) or isinstance(child, nn.Dropout2d) or isinstance(child, nn.Dropout3d):
            setattr(module, name, nn.Identity())
        else:
            remove_dropout(child)


def find_hidden_state_size(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            return min(module.weight.shape)
    return None


@torch.no_grad()
def reinit_lora_modules(name, module, init_config, peft_conf, **kwargs):
    r"""
    Reinitialize the lora model with the given configuration.
    """
    lora_r = min(module.lora_A.default.weight.shape)
    a_dim = max(module.lora_A.default.weight.shape)
    b_dim = max(module.lora_B.default.weight.shape)
    if init_config.mode == "simple":
        match init_config.lora_A:
            case "gaussian":
                torch.nn.init.normal_(
                    module.lora_A.default.weight, mean=0.0, std=init_config.lora_A_std
                )
            case "kaiming":
                # https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L124
                torch.nn.init.kaiming_uniform_(module.lora_A.default.weight, a=math.sqrt(5))
            case "fan_out_kaiming":
                torch.nn.init.kaiming_normal_(
                    module.lora_A.default.weight, mode="fan_out"
                )
            case "xavier":
                torch.nn.init.xavier_normal_(module.lora_A.default.weight)
            case "zeros":
                torch.nn.init.zeros_(module.lora_A.default.weight)
            case "unit":
                torch.nn.init.normal_(
                    module.lora_A.default.weight, mean=0.0, std=1.0 / (a_dim**0.5)
                )
            case "orthogonal":
                torch.nn.init.orthogonal_(module.lora_A.default.weight)
            case _:
                raise ValueError(f"Unknown lora_A initialization: {init_config.lora_A}")
        match init_config.lora_B:
            case "gaussian":
                torch.nn.init.normal_(
                    module.lora_B.default.weight, mean=0.0, std=init_config.lora_B_std
                )
            case "kaiming":
                torch.nn.init.kaiming_normal_(module.lora_B.default.weight)
            case "fan_out_kaiming":
                torch.nn.init.kaiming_normal_(
                    module.lora_B.default.weight, mode="fan_out"
                )
            case "xavier":
                torch.nn.init.xavier_normal_(module.lora_B.default.weight)
            case "zeros":
                torch.nn.init.zeros_(module.lora_B.default.weight)
            case "unit":
                torch.nn.init.normal_(
                    module.lora_B.default.weight, mean=0.0, std=1.0 / (b_dim**0.5)
                )
            case "orthogonal":
                torch.nn.init.orthogonal_(module.lora_B.default.weight)
            case _:
                raise ValueError(f"Unknown lora_B initialization: {init_config.lora_B}")
        if init_config.get("scale", "") == "stable":
            gamma = init_config.stable_gamma
            module.lora_B.default.weight.data *= (m**0.25) / gamma**0.5
            module.lora_A.default.weight.data *= (n**0.25) / gamma**0.5
    elif init_config.mode == "svd":
        U, S, V = torch.svd_lowrank(module.weight.float(), q=4 * lora_r, niter=4)
        V = V.T
        m, n = module.weight.shape
        if init_config.scale == "default":
            S = S / module.scaling["default"]
            module.lora_B.default.weight = torch.nn.Parameter(
                (U[:, :lora_r] * torch.sqrt(S[:lora_r])).contiguous()
            )
            module.lora_A.default.weight = torch.nn.Parameter(
                (V[:lora_r, :].T * torch.sqrt(S[:lora_r])).T.contiguous()
            )
        elif init_config.scale == "stable":
            gamma = init_config.stable_gamma
            module.lora_B.default.weight = torch.nn.Parameter(
                (U[:, :lora_r] * (m**0.25) / gamma**0.5).contiguous()
            )
            module.lora_A.default.weight = torch.nn.Parameter(
                (V[:lora_r, :] * (n**0.25) / gamma**0.5).contiguous()
            )
        elif init_config.scale == "unit":
            module.lora_B.default.weight = torch.nn.Parameter(
                (U[:, :lora_r]).contiguous()
            )
            module.lora_A.default.weight = torch.nn.Parameter(
                (V[:lora_r, :]).contiguous()
            )
        elif init_config.scale == "normalized":
            S_sum = S[:lora_r].sum()
            module.lora_B.default.weight = torch.nn.Parameter(
                (U[:, :lora_r] * torch.sqrt(S[:lora_r])/torch.sqrt(S_sum)*lora_r**0.5).contiguous()
            )
            module.lora_A.default.weight = torch.nn.Parameter(
                (V[:lora_r, :].T * torch.sqrt(S[:lora_r])/torch.sqrt(S_sum)*lora_r**0.5).T.contiguous()
            )
    elif init_config.mode == "gradient":
        named_grad = kwargs["named_grads"]
        if init_config.direction == "LoRA-FA":
            grad_name = ".".join(name.split(".")[2:])
        else:
            grad_name = ".".join(name.split(".")[2:]) + ".weight"
        grads = named_grad[grad_name]
        if init_config.direction == "LoRA-One" or init_config.direction == "LoRA-FA":
            U, S, V = torch.svd_lowrank(-grads.cuda().float(), q=512, niter=16) #(d_out × k), (k,), (d_in × k)
        else:
            U, S, V = torch.svd_lowrank(grads.cuda().float(), q=512, niter=16)
        V = V.T
        if init_config.direction == "LoRA-FA":
            inv_sqrt_C = kwargs["inv_sqrt_C"][grad_name].to(V.device) # d_out X d_out
            inv_sqrt_Sigma_X = kwargs["inv_sqrt_Sigma_X"][grad_name].to(V.device)  # d_in X d_in
            BA = inv_sqrt_C @ U[:, :lora_r] @ torch.diag(S[:lora_r]) @ V[:lora_r, :] @ inv_sqrt_Sigma_X 
            # print('[LoRA-FA] ', BA.shape)
            U1, S1, V1 = torch.svd_lowrank(BA.cuda().float(), q=512, niter=16)
            V1 =V1.T
            B = U1[:, :lora_r] @ torch.diag(torch.sqrt(S1[:lora_r])) / torch.sqrt(S1[0])
            A = torch.diag(torch.sqrt(S1[:lora_r])) @ V1[:lora_r, :] / torch.sqrt(S1[0])
        elif init_config.direction == "LoRA-One":
            B = U[:, :lora_r] @ torch.diag(torch.sqrt(S[:lora_r])) / torch.sqrt(S[0])
            A = torch.diag(torch.sqrt(S[:lora_r])) @ V[:lora_r, :] / torch.sqrt(S[0])
        elif init_config.direction == "LoRA-GA":
            B = U[:, lora_r : 2 * lora_r]
            A = V[:lora_r, :]
        scaling_factor = module.scaling["default"]
        if init_config.scale == "gd":
            A = A / scaling_factor
            B = B / scaling_factor
        elif init_config.scale == "unit":
            # Because A,B is orthogonal, do not need to scale
            pass
        elif init_config.scale == "stable":
          if init_config.direction == "LoRA-One" or init_config.direction == "LoRA-FA":
            gamma = init_config.stable_gamma
            B = B / gamma**0.5
            A = A / gamma**0.5
          else:
            m, n = grads.shape # m: feature_out, n: feature_in
            # the scale of output is only related to the feature_out
            gamma = init_config.stable_gamma
            B = B * m**0.25 / gamma**0.5
            A = A * m**0.25 / gamma**0.5
        elif init_config.scale == "weightS":
            _, S, _ = torch.svd_lowrank(module.weight.float(), q=4 * lora_r, niter=4)
            S = S / module.scaling["default"]
            avg_s = torch.sqrt(S[:lora_r]).mean().to(A.device)
            B = B * avg_s
            A = A * avg_s

        # construct new magnitude vectors if use DoRA
        if peft_conf.get("dora", False):
           # temp matrix
           V = module.weight.float() + (peft_conf.lora_alpha/math.sqrt(lora_r)) * B @ A
           mag_vec = torch.norm(V, p=2, dim=1)
        else:
           pass        

        module.lora_B.default.weight = torch.nn.Parameter(B.contiguous().cuda())
        module.lora_A.default.weight = torch.nn.Parameter(A.contiguous().cuda())
        if peft_conf.get("dora", False):
           module.lora_magnitude_vector.default.weight = torch.nn.Parameter(mag_vec.contiguous().cuda())
        # print(f"[{name}] Expected A={module.lora_A.default.weight.shape}, "f"B={module.lora_B.default.weight.shape}, "f"Got A={A.shape}, B={B.shape}") #TODO
    with torch.no_grad():
        if peft_conf.get("dora", False): #DoRA uses fp16
                module.lora_A.default.weight.data = module.lora_A.default.weight.data.to(
                    torch.float16
                )
                module.lora_B.default.weight.data = module.lora_B.default.weight.data.to(
                    torch.float16
                )
                module.lora_magnitude_vector.default.weight.data = module.lora_magnitude_vector.default.weight.data.to(
                    torch.float16
                )
        else:
            # consider dtype not in init_config
            if "dtype" not in init_config:
                pass
            elif init_config.dtype == "bf16":
                module.lora_A.default.weight.data = module.lora_A.default.weight.data.to(
                    torch.bfloat16
                )
                module.lora_B.default.weight.data = module.lora_B.default.weight.data.to(
                    torch.bfloat16
                )
            elif init_config.dtype == "fp32":
                module.lora_A.default.weight.data = module.lora_A.default.weight.data.to(
                    torch.float32
                )
                module.lora_B.default.weight.data = module.lora_B.default.weight.data.to(
                  torch.float32
                )

        # If lora_A@lora_B is not zero, then we need to subtract lora_A@lora_B from the original weight matrix
        if init_config.direction == "LoRA-One" or init_config.direction == "LoRA-FA":
          pass
        else:
          offset = (module.lora_B.default.weight @ module.lora_A.default.weight).to(
              module.weight.data.device
          )
          scaling_factor = module.scaling["default"]
          offset *= scaling_factor
          if "norm_clip" in init_config and init_config.norm_clip:
              # for numerical stability, offset's largest value must be less then weight's largest value
              ratio = torch.max(torch.abs(module.weight.data)) / torch.max(
                  torch.abs(offset)
              )
              if ratio < 1:
                  offset *= ratio
                  module.lora_A.default.weight.data *= ratio**0.5
                  module.lora_B.default.weight.data *= ratio**0.5
                  log.warning(f"Clipping offset by {ratio}")
          try:
              module.weight.data -= offset
          except:
              breakpoint()


def reinit_lora(model, init_config, peft_conf, **kwargs):
    r"""
    Reinitialize the lora model with the given configuration.
    """
    for name, module in tqdm(
        model.named_modules(),
        desc="Reinitializing Lora",
        total=len(list(model.named_modules())),
    ):
        if isinstance(module, LoraLinear):
            reinit_lora_modules(name, module, init_config, peft_conf, **kwargs)

    return model

@contextmanager
def temporary_weights(model, adapted_weights, eta):
    r"""
    Context manager for temporarily applying weight deltas
    """
    # Save original weights
    original_params = {
        name: param.data.clone()
        for name, param in model.named_parameters()
        if name in adapted_weights
    }
    
    # Apply adapted weights
    for name, param in model.named_parameters():
        if name in adapted_weights:
            param.data -= eta * adapted_weights[name].to(param.device)
    
    try:
        yield model  # Return test model
    finally:
        # Restore original weights
        for name, param in model.named_parameters():
            if name in original_params:
                param.data.copy_(original_params[name])

def search_eta(model, dataset, batch_size, **kwargs):
    r"""
    Search optimal scaling eta with the given grid of values.
    """
    eta_list = [10.0, 5.0, 1.0, 5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
    min_loss = float('inf')
    best_eta = None

    named_grad = kwargs["named_grads"]
    for eta in tqdm(eta_list):
        with temporary_weights(model, named_grad, eta):
             # Model now has original - eta * adapted weights
             model.train()
             dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
             num = 0
             loss = 0
             for batch in tqdm(dataloader, desc="Computing loss"):
                 num += 1
                 batch = {k: v.to(model.device) for k, v in batch.items()}
                 outputs = model(**batch)
                 loss += outputs.loss.item()
                 for n, p in model.named_parameters():
                    if p.grad is not None:
                       p.grad = None

             loss /= num
             print(f"Temporary loss: {loss} for eta= {eta}")

             if loss < min_loss:
                min_loss = loss
                best_eta = eta

             torch.cuda.empty_cache()

    return best_eta

#grad_name = ".".join(name.split(".")[2:]) + ".weight"
#grads = named_grad[grad_name]

def get_record_gradient_hook(model, record_dict):
    def record_gradient_hook(grad):
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                if n not in record_dict:
                    record_dict[n] = p.grad.cpu()
                else:
                    record_dict[n] += p.grad.cpu()
                p.grad = None
        return grad

    return record_gradient_hook


def estimate_gradient(
    model, dataset, batch_size: int = 4
) -> Dict[str, List[torch.Tensor]]:
    r"""
    Estimate the gradient of the model on the given dataset
    """
    log.info("Estimating gradient")
    model.train()
    named_grads = {}
    hooks = []
    for name, param in model.named_parameters():
        hook = param.register_hook(get_record_gradient_hook(model, named_grads))
        hooks.append(hook)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    num = 0
    for batch in tqdm(dataloader, desc="Estimating gradient"):
        num += 1
        batch = {k: v.to(model.device) for k, v in batch.items()}
        outputs = model(**batch)
        outputs.loss.backward()
        get_record_gradient_hook(model, named_grads)(None)  # just for the safe side, they are calculating the gradient for entire params. # get gradient of last layer
        # make sure the gradient is cleared
        for n, p in model.named_parameters():
            if p.grad is not None:
                p.grad = None
    for n, g in named_grads.items():
        named_grads[n] /= num
    for hook in hooks:
        hook.remove()
    torch.cuda.empty_cache()
    return named_grads


def estimate_dataset_whitened_H_and_inv_roots(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    lora_target_modules: List[str],
    batch_size: int = 4,
    reg_alpha: float = 1e-6,
    epsilon: float = 1e-8,
    use_cholesky: bool = False,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Dataset-level statistically-correct estimator.
    For each target Linear module (identified by substrings in `lora_target_modules`) this function returns:
        - tilde_H:  C^{1/2} @ ( avg_over_samples[J X^T] ) @ Sigma_X^{1/2}
        - inv_sqrt_C: (C)^{-1/2}  (dataset-level)
        - inv_sqrt_Sigma_X: (Sigma_X)^{-1/2}  (dataset-level)
    Implementation details:
        - For each batch, we flatten samples across batch and sequence dims to form:
            J_flat: (num_samples, hidden_dim)
            X_flat: (num_samples, input_dim)
        - We accumulate C_full = sum(J_flat^T @ J_flat), Sigma_X_full = sum(X_flat^T @ X_flat), cross_full = sum(J_flat^T @ X_flat).
        - At the end we form averages (divide by num_of_batch), compute matrix roots/inverse roots once,
          build tilde_H, and return (tilde_H, inv_sqrt_C, inv_sqrt_Sigma_X).
    Note: The function keeps all computations on `device` when possible.
    """
    model.train()
    device = model.device
    C_full: Dict[str, torch.Tensor] = {}
    Sigma_X_full: Dict[str, torch.Tensor] = {}
    cross_full: Dict[str, torch.Tensor] = {}
    total_samples: Dict[str, int] = {}
    hooks = []

    def compute_matrix_roots(mat: torch.Tensor, use_cholesky: bool = False, eps: float = epsilon):
        # assume mat on correct device
        if use_cholesky:
            mat = mat + eps * torch.eye(mat.shape[0], device=mat.device, dtype=mat.dtype)
            # cholesky will raise if not PD, but eps should help
            L = torch.linalg.cholesky(mat)
            sqrt_mat = L
            inv_L = torch.linalg.inv(L)
            inv_sqrt_mat = inv_L.T @ inv_L
        else:
            eigvals, eigvecs = torch.linalg.eigh(mat)
            eigvals = torch.clamp(eigvals, min=eps)
            sqrt_mat = eigvecs @ torch.diag(torch.sqrt(eigvals)) @ eigvecs.T
            inv_sqrt_mat = eigvecs @ torch.diag(1.0 / torch.sqrt(eigvals)) @ eigvecs.T
        return sqrt_mat, inv_sqrt_mat

    for name, module in tqdm(model.named_modules(), desc=f"Attaching hooks to {lora_target_modules} layers"):
        if isinstance(module, torch.nn.Linear) and any(t in name for t in lora_target_modules):
            def forward_hook(mod, inputs, output, lname=name):
                mod._saved_input_for_lora = inputs[0]

            def backward_hook(mod, grad_input, grad_output, lname=name):
                J = grad_output[0].detach()
                X = getattr(mod, "_saved_input_for_lora", None).detach()
                Xf = X.reshape(-1, X.shape[-1]).to(dtype=torch.float32, device=device)
                Jf = J.reshape(-1, J.shape[-1]).to(dtype=torch.float32, device=device)
                ns = Xf.shape[0]

                C_batch = Jf.T @ Jf                 # (hidden_dim x hidden_dim)
                Sigma_X_batch = Xf.T @ Xf           # (input_dim x input_dim)
                cross_batch = Jf.T @ Xf             # (hidden_dim x input_dim) #TODO multiply by eta
                if lname not in cross_full:
                    C_full[lname] = C_batch.clone()
                    Sigma_X_full[lname] = Sigma_X_batch.clone()
                    cross_full[lname] = cross_batch.clone()
                    total_samples[lname]= ns
                else:
                    C_full[lname] += C_batch
                    Sigma_X_full[lname] += Sigma_X_batch
                    cross_full[lname] += cross_batch
                    total_samples[lname] +=ns
                try:
                    delattr(mod, "_saved_input_for_lora")
                except Exception:
                    pass
            hooks.append(module.register_forward_hook(forward_hook))
            hooks.append(module.register_full_backward_hook(backward_hook))

    if not hooks:
        print("[LoRA-FA] Warning: no target modules found for lora targets:", lora_target_modules)
        return {}

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    for batch in tqdm(dataloader, desc="Accumulating dataset-level outer products"):
        model.zero_grad(set_to_none=True)
        try:
            batch = {k: v.to(model.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            outputs = model(**batch)
            outputs.loss.backward()
        except Exception as e:
            print(f"[LoRA-FA] Skipping batch due to error: {e}")
            continue
    for h in hooks:
        h.remove()

    result_tilde_H: Dict[str, torch.Tensor] = {}
    inv_sqrt_C: Dict[str, torch.Tensor] = {}
    inv_sqrt_Sigma_X: Dict[str, torch.Tensor] = {}
    # reg_alpha = 1e-6
    print('[LoRA-FA] Using reg_alpha: ', reg_alpha)
    for lname in cross_full.keys():
        # print(f'[LoRA-FA] total_samples in {lname}::', total_samples[lname])
        hidden_dim = C_full[lname].shape[0]
        input_dim = Sigma_X_full[lname].shape[0]
        C_avg = C_full[lname] / float(total_samples[lname]) + reg_alpha * torch.eye(hidden_dim, device=device)
        Sigma_X_avg = Sigma_X_full[lname] / float(total_samples[lname]) + reg_alpha * torch.eye(input_dim, device=device)

        # C_avg = ((1 - cons1) * C_full[lname] / float(total_samples[lname])+ cons1 * torch.eye(hidden_dim, device=device)) 
        # Sigma_X_avg = ((1 - cons2) * Sigma_X_full[lname] / float(total_samples[lname]) + cons2 * torch.eye(input_dim, device=device)) 

        # C_avg = C_full[lname] / float(total_samples[lname])           # (hidden_dim x hidden_dim)
        # Sigma_X_avg = Sigma_X_full[lname] / float(total_samples[lname])  # (input_dim x input_dim)
        cross_avg = cross_full[lname] / float(total_samples[lname])   # (hidden_dim x input_dim) approximates J X^T / N

        # compute sqrt and inv sqrt from dataset-level covariances
        sqrt_C, invC = compute_matrix_roots(C_avg, use_cholesky=use_cholesky, eps=epsilon)
        sqrt_Sigma, invSigma = compute_matrix_roots(Sigma_X_avg, use_cholesky=use_cholesky, eps=epsilon)

        result_tilde_H[lname] = sqrt_C @ cross_avg @ sqrt_Sigma 
        inv_sqrt_C[lname] = invC
        inv_sqrt_Sigma_X[lname] = invSigma

    torch.cuda.empty_cache()
    return {
        "tilde_H": result_tilde_H,
        "inv_sqrt_C": inv_sqrt_C,
        "inv_sqrt_Sigma_X": inv_sqrt_Sigma_X,
    }


@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def run_exp(cfg: DictConfig):
    log.info(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.seed)
    model_name = cfg.model.name
    model_type = cfg.model.type
    dataset_name = cfg.dataset_name
    if dataset_name in ['lamp_news_headlines', 'lamp_scholarly_titles', 'longlamp_abstract_generation', 'longlamp_product_review', 'longlamp_topic_writing']:
        dataset_func = DATASET_MAP['persona']
    else:
        dataset_func = DATASET_MAP[dataset_name]
    use_peft = cfg.peft.use_peft
    if_use_rslora = cfg.peft.use_rslora
    lora_r = cfg.peft.lora_r
    lora_relative_r = cfg.peft.lora_relative_r
    lora_target_modules = cfg.peft.lora_target_modules
    train_embeddings = cfg.peft.train_embeddings

    accelerator = Accelerator()

    if cfg.dry_run:
        return
    if use_peft:
        assert (lora_r is not None) ^ (
            lora_relative_r is not None
        ), "Please specify lora_r or lora_relative_r"
        assert lora_target_modules is not None, "Please specify lora_target_modules"
    else:
        lora_r = None
        lora_target_modules = None
        lora_relative_r = None
        train_embeddings = True
    config = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "use_peft": use_peft,
        "lora_r": lora_r,
        "lora_target_modules": str(lora_target_modules),
        "lora_relative_r": lora_relative_r,
        "train_embeddings": train_embeddings,
    }
    if cfg.wandb.name:
        name = cfg.wandb.name
    else:
        name = "_".join([f"{k}={v}" for k, v in config.items()])
    cfg.wandb.project += "_" + cfg.dataset_name
    wandb.init(
        project=cfg.wandb.project,
        name=name,
        config=config,
    )
    if dataset_name in ['lamp_news_headlines', 'lamp_scholarly_titles', 'longlamp_abstract_generation', 'longlamp_product_review', 'longlamp_topic_writing']:
        train_set, val_set, _ = dataset_func(dataset_name)
    else:
        train_set, val_set, _ = dataset_func()
    if 'test' in name: 
        print("[TEST] Actual size of train_set: ", len(train_set))
        train_set = train_set.select(range(100))
        val_set = val_set.select(range(10))
    model, tokenizer = initialize_text_to_text_model(
        model_name, model_type, cfg.model.bf16, cfg.peft.use_peft, flash_attention=True
    ) #From here, the pretrained model is initialized

    model = model.to('cuda')

    additional_kwargs = {} #generate empty args

    if lora_target_modules == "all":
        lora_target_modules = find_all_linear_modules(model) 
    elif lora_target_modules == "last":
        lora_target_modules = get_llama_last_layers(model) 
    else:
        lora_target_modules = list(lora_target_modules) if lora_target_modules else []
    print('LoRA Target Modules ', lora_target_modules)
    if use_peft and cfg.init.mode == "gradient":
        if isinstance(train_set, list):
            temp_set = train_set[: cfg.init.bsz * cfg.init.iters]
        else:
            temp_set = train_set.select(range(cfg.init.bsz * cfg.init.iters))
            print('Batch size for gradient calculation', len(temp_set))
        transform_dataset(
            model_type=model_type,
            dataset=temp_set,
            tokenizer=tokenizer,
            max_length=cfg.init.max_length,
        )
        if cfg.init.direction == 'LoRA-FA':
            # remove_dropout(model) #TODO
            estimates = estimate_dataset_whitened_H_and_inv_roots(model, temp_set, lora_target_modules, cfg.init.bsz, cfg.init.reg_alpha)
            additional_kwargs["named_grads"] = estimates['tilde_H']
            additional_kwargs["inv_sqrt_C"] = estimates['inv_sqrt_C']
            additional_kwargs["inv_sqrt_Sigma_X"]  = estimates['inv_sqrt_Sigma_X']
        else:
            # remove_dropout(model) #TODO
            named_grads = estimate_gradient(model, temp_set, cfg.init.bsz)
            additional_kwargs["named_grads"] = named_grads #append grads
            #From here, we got full-batch GD gradients
            #best_eta = search_eta(model, temp_set, cfg.init.bsz, **additional_kwargs)
            #additional_kwargs["eta"] = best_eta
    if lora_relative_r is not None:
        hidden_size = find_hidden_state_size(model)
        lora_r = int(hidden_size * lora_relative_r)
        log.info(f"lora_r is set to {hidden_size} * {lora_relative_r} = {lora_r}")
    if use_peft and cfg.peft.get("dora", False):
        log.info("Using Dora")
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=cfg.peft.lora_alpha,
            lora_dropout=cfg.peft.lora_dropout,
            target_modules=lora_target_modules,
            use_rslora=if_use_rslora,
            use_dora=True,
        )
        orig_model_params = sum(p.numel() for p in model.parameters())
        model = get_peft_model(model, peft_config)
        ###############################################Re-init DoRA if using LoRA-One
        if cfg.init.mode == "gradient":
            log.info("Initializing DoRA-One")
            reinit_lora(model, cfg.init, cfg.peft, **additional_kwargs)
        trainable_params, all_param = model.get_nb_trainable_parameters()
        rate = {
            "trainable_params": trainable_params,
            "orig_params": orig_model_params,
            "all_params": all_param,
            "trainable_ratio": trainable_params / all_param,
            "param_ratio": trainable_params / orig_model_params,
        }
    elif use_peft and cfg.peft.get("adalora", False):
        log.info("Using AdaLora")
        peft_config = AdaLoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_r=lora_r,
            lora_alpha=cfg.peft.lora_alpha,
            target_modules=lora_target_modules,
            total_step=int(len(train_set)/cfg.model.real_batch_size)*cfg.model.epochs,
        )
        orig_model_params = sum(p.numel() for p in model.parameters())
        model = get_peft_model(model, peft_config)
        trainable_params, all_param = model.get_nb_trainable_parameters()
        rate = {
            "trainable_params": trainable_params,
            "orig_params": orig_model_params,
            "all_params": all_param,
            "trainable_ratio": trainable_params / all_param,
            "param_ratio": trainable_params / orig_model_params,
        }
    elif use_peft: # Reinit LoRA here
        if cfg.init.mode == "gradient":
           peft_config = LoraConfig(
               r=lora_r,
               lora_alpha=cfg.peft.lora_alpha, # cancel square root of lora rank if needed
               lora_dropout=cfg.peft.lora_dropout,
               target_modules=lora_target_modules,
               use_rslora=if_use_rslora,
           )
        else:
           peft_config = LoraConfig(
               r=lora_r,
               lora_alpha=cfg.peft.lora_alpha,
               lora_dropout=cfg.peft.lora_dropout,
               target_modules=lora_target_modules,
               use_rslora=if_use_rslora,
           )
        orig_model_params = sum(p.numel() for p in model.parameters())
        ########## We need to determine scaling parameter here
        model = get_peft_model(model, peft_config)
        reinit_lora(model, cfg.init, cfg.peft, **additional_kwargs)
        if train_embeddings:
            model.lm_head.weight.requires_grad = True
        trainable_params, all_param = model.get_nb_trainable_parameters()
        rate = {
            "trainable_params": trainable_params,
            "orig_params": orig_model_params,
            "all_params": all_param,
            "trainable_ratio": trainable_params / all_param,
            "param_ratio": trainable_params / orig_model_params,
        }
        if cfg.init.mode == "gradient":
            if cfg.init.direction != "LoRA-One" and cfg.init.direction != "LoRA-FA":
              save_dir = os.path.join(
                  "results", f"{cfg.wandb.project}/{name}/{cfg.seed}", "orig_checkpoint"
              )
              model.save_pretrained(save_dir)
              adapter_config = json.load(open(os.path.join(save_dir, "adapter_config.json")))
              adapter_config["lora_alpha"] = -adapter_config["lora_alpha"]
              json.dump(
                  adapter_config, open(os.path.join(save_dir, "adapter_config.json"), "w")
              )
    else:
        # full finetune
        all_param = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        rate = {
            "trainable_params": trainable_params,
            "orig_params": all_param,
            "all_params": all_param,
            "trainable_ratio": trainable_params / all_param,
            "param_ratio": 1,
        }
    log.info(rate)
    # log rate into wandb summary
    wandb.summary.update(rate)
    training_loop = train_text_to_text_model
    model = training_loop(
        f"{cfg.wandb.project}/{name}",
        train_set,
        val_set,
        model,
        tokenizer,
        model_type,
        optimizer=None, # using custom_optimizer
        num_train_epochs=cfg.model.epochs,
        per_device_batch_size=cfg.model.per_device_batch_size,
        real_batch_size=cfg.model.real_batch_size,
        bf16=cfg.model.bf16,
        eval_epochs=cfg.model.eval_epochs,
        early_stopping_patience=cfg.model.early_stopping_patience,
        max_length=cfg.model.max_length,
        logging_steps=cfg.model.logging_steps,
        use_loraplus=cfg.peft.use_loraplus,
        loraplus_lr_ratio=cfg.peft.loraplus_lr_ratio,
        learning_rate=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
        lr_scheduler_type = cfg.model.lr_scheduler_type,
        warmup_ratio=cfg.model.warmup_ratio,
        warmup_steps=cfg.model.warmup_steps,
        num_process=accelerator.num_processes,
        gradient_checkpointing=cfg.get("gradient_checkpointing", False),
        seed=cfg.seed,
    )
    save_dir = os.path.join(
        "results", f"{cfg.wandb.project}/{name}/{cfg.seed}", "merged_checkpoint"
    )
    if not use_peft:
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
    else:
      if not cfg.model.saving:
        pass
      else:
        merge_llama(os.path.join("results", f"{cfg.wandb.project}/{name}/{cfg.seed}")) # if lamma
        #merge_t5(os.path.join("results", f"{cfg.wandb.project}/{name}/{cfg.seed}"))
    log.info(f"Saving model to {save_dir}")

    save_safe_dir = os.path.join(
        "safe_results", f"{cfg.wandb.project}/{name}/{cfg.seed}", "final_checkpoint"
    )
    model.save_pretrained(save_safe_dir)
    log.info(f"Saving safe adapters to {save_safe_dir} for copies")

    '''orig_model, _ = initialize_text_to_text_model(
        model_name, model_type, cfg.model.bf16, cfg.peft.use_peft, flash_attention=True
    )

    finetuned_model, _ = initialize_text_to_text_model(
        save_dir, model_type, cfg.model.bf16, cfg.peft.use_peft, flash_attention=True
    )

    for (name_f, param_f), (name_p, param_p) in zip(
            finetuned_model.named_parameters(),
            orig_model.named_parameters()
    ):
      if param_f.ndim == 2 and min(param_f.shape) > 1:
        if name_f != name_p:
          log.info(f"{name_f} mismatched {name_p}")
          continue
        
        param_f_data = param_f.data.type(torch.float32).cuda()
        param_p_data = param_p.data.type(torch.float32).cuda()
        diff = param_f_data - param_p_data

        grads = named_grads[name_p]
        U, _, V = torch.svd_lowrank(grads.cuda().float(), q=4 * lora_r, niter=4)
        P, _, Q = torch.svd_lowrank(diff, q=4 * lora_r, niter=4)

        principal_angle_a = torch.svd(U[:,lora_r:].t() @ P[:,:lora_r]).S.max()
        principal_angle_b = torch.svd(V[:,lora_r:].t() @ Q[:,:lora_r]).S.max()

        log.info(f"Principal angle of A for {name_f}: {principal_angle_a}")
        log.info(f"Principal angle of B for {name_f}: {principal_angle_b}")'''

    wandb.finish()


if __name__ == "__main__":
    run_exp()
