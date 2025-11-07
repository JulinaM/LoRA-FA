# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import copy
import json
import os
import re
import sys
import argparse
from utils import initialize_text_to_text_model, load_peft_model
import torch
from tqdm import tqdm
from transformers import GenerationConfig
import wandb
from datasets import Dataset

# Determine device
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass


# ------------------- INFERENCE FUNCTION -------------------
@torch.inference_mode()
def infer_commonsense(model, tokenizer, dataset, dataset_name, batch_size=4):
    """Efficient batched inference for commonsense reasoning datasets."""
    def _predict_commonsense(examples):
        prompts = [generate_prompt(inst) for inst in examples["instruction"]]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512, add_special_tokens=False)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # Primary generation configuration
        generation_config = GenerationConfig(
            do_sample=True,
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            num_beams=4,
            max_new_tokens=32,
        )

        try:
            outputs = model.generate(
                **inputs,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
            )
        except RuntimeError:
            print("⚠️ Sampling failed, switching to greedy generation.")
            greedy_config = GenerationConfig(
                do_sample=False,
                num_beams=1,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                max_new_tokens=32,
            )
            outputs = model.generate(
                **inputs,
                generation_config=greedy_config,
                return_dict_in_generate=True,
                output_scores=True,
            )

        decoded = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
        preds = [o.split("### Response:")[-1].strip() for o in decoded]
        return {"prediction": preds}

    predictions = dataset.map(
        lambda x: _predict_commonsense(x),
        batched=True,
        batch_size=batch_size,
    )["prediction"]

    references = dataset["answer"]
    return predictions, references


# ------------------- MAIN PIPELINE -------------------
def main():
    args = parse_args()

    wandb.init(project="commonsense_eval", name=args.wandb_name)

    dataset_list = load_data(args)
    dataset = Dataset.from_list(dataset_list)

    model_type = "CausalLM"
    model, tokenizer = initialize_text_to_text_model(
        "meta-llama/Llama-2-7b-hf",
        model_type,
        True,
        flash_attention=True
    )

    model = load_peft_model(model, f'./results/lorafa_commonsense_reasoning/{args.wandb_name}/9/')
    model = model.to(device)
    model.half()

    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = 0
    model.config.pad_token_id = 0
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    print("🚀 Running inference...")
    predictions, references = infer_commonsense(model, tokenizer, dataset, args.dataset, args.batch_size)

    print("✅ Extracting predicted labels...")
    preds_extracted = [extract_answer(args, p) for p in predictions]

    correct = sum([pred.lower().strip() == ref.lower().strip() for pred, ref in zip(preds_extracted, references)])
    acc = correct / len(references)
    print(f"\n🎯 Final Accuracy: {acc:.4f}")

    # Save + log results
    os.makedirs('eval_results/commonsense_eval', exist_ok=True)
    save_file = f'eval_results/commonsense_eval/{args.dataset}_{args.wandb_name}.txt'

    with open(save_file, "a") as f:
        f.write(f"{args.wandb_name}: {args.dataset} {acc:.4f}\n")

    wandb.log({f"{args.dataset}/Acc": acc})
    print("✅ Evaluation complete and logged.")


# ------------------- HELPERS -------------------

template_wo_input = '''Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
'''

def generate_prompt(instruction, input=None):
    return template_wo_input.format(instruction=instruction)


def load_data(args):
    file_path = f'../hf-datasets/commonsense_eval/{args.dataset}/test.json'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cannot find dataset file: {file_path}")
    json_data = json.load(open(file_path, 'r'))
    return json_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=[
        "boolq", "piqa", "social_i_qa", "hellaswag",
        "winogrande", "ARC-Challenge", "ARC-Easy", "openbookqa"
    ], required=True)
    parser.add_argument('--wandb_name', required=True)
    parser.add_argument('--batch_size', type=int, default=4)
    return parser.parse_args()


def extract_answer(args, sentence: str) -> str:
    dataset = args.dataset
    sentence_ = sentence.strip().lower()

    if dataset == 'boolq':
        match = re.findall(r'true|false', sentence_)
    elif dataset == 'piqa':
        match = re.findall(r'(solution1|solution2)', sentence_, re.IGNORECASE)
    elif dataset in ['social_i_qa', 'ARC-Challenge', 'ARC-Easy', 'openbookqa']:
        match = re.findall(r'answer[1-5]', sentence_)
    elif dataset == 'hellaswag':
        match = re.findall(r'ending[1-4]', sentence_)
    elif dataset == 'winogrande':
        match = re.findall(r'option[1-2]', sentence_)
    else:
        match = []

    return match[0] if match else ""


# ------------------- ENTRY POINT -------------------
if __name__ == "__main__":
    main()
