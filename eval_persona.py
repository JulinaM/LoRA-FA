#!/usr/bin/env python3
# personalized_hf_infer.py
# Usage:
#   python personalized_hf_infer.py \
#       --base_model llama
#       --wandb_name loraone-8-V1-llama \
#       --adapter_dir ./results/lorafa_personalized/loraone-8-V1-llama/42/ \
#       --data /path/to/data.jsonl \
#       --out_dir ./outputs \
#       --max_tokens 512

import argparse
import json
import os
import sys
from typing import List, Dict, Any
import torch
from tqdm import tqdm

# === import your utils ===
from utils import model_inference, initialize_text_to_text_model, load_peft_model


# ------------------------------
# Dataset loading (same as before)
# ------------------------------
def load_dataset(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if "\n" in text:
        lines = [ln for ln in text.splitlines() if ln.strip()]
        try:
            return [json.loads(ln) for ln in lines]
        except json.JSONDecodeError:
            pass
    js = json.loads(text)
    if isinstance(js, dict):
        return [js]
    if isinstance(js, list):
        return js
    raise ValueError("Unsupported dataset format: expected JSONL, JSON list, or JSON object.")


def make_prompt(item: Dict[str, Any]) -> str:
    instr = item.get("instruction", "")
    inp = item.get("input", "")
    if inp:
        return f"{instr}\n{inp}"
    return instr


def get_ground_truth(item: Dict[str, Any]) -> str:
    output = str(item.get("output", ""))
    if output.startswith("### Response: "):
        output = output[len("### Response: "):]
    return output


# ------------------------------
# Main Inference Logic (HuggingFace + LoRA)
# ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True, help="Base HF model path or repo (e.g., meta-llama/Llama-2-7b-hf)")
    ap.add_argument("--wandb_name", required=True, help="WandB experiment name prefix (used in adapter path)")
    ap.add_argument("--adapter_dir", required=True, help="Path to fine-tuned LoRA adapter directory")
    ap.add_argument("--data", required=True, help="Path to dataset (json/jsonl)")
    ap.add_argument("--out_dir", required=True, help="Directory to save outputs")
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load dataset
    data = load_dataset(args.data)
    if len(data) == 0:
        print("Empty dataset.", file=sys.stderr)
        sys.exit(1)

    # Build prompts and ground truths
    prompts, ground_truths = [], []
    for item in data:
        prompts.append(make_prompt(item))
        ground_truths.append(get_ground_truth(item))

    # ------------------------------
    # Load model + adapter
    # ------------------------------
    print(f"Loading base model: {args.base_model}")
    base_model = "meta-llama/Llama-2-7b-hf" if args.base_model == "llama" else "mistralai/Mistral-7B-Instruct-v0.2"
    print("using model", base_model)
    model, tokenizer = initialize_text_to_text_model(
        base_model,
        "CausalLM",
        True,
        flash_attention=True
    )
    print(f"Loading adapter from: {args.adapter_dir}")
    model = load_peft_model(model, args.adapter_dir)
    model = model.to(args.device)
    model.eval()

    # ------------------------------
    # Inference
    # ------------------------------
    print(f"Generating {len(prompts)} responses using LoRA model ...")

    results = []
    for i, prompt in enumerate(tqdm(prompts)):
        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                temperature=0.0,
                top_k=10,
                top_p=1.0,
                repetition_penalty=1.15,
                do_sample=False,
            )
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # remove the prompt prefix if the model echoes input
            generated_text = generated_text.replace(prompt, "").strip()

        rec = {
            "prompt": prompt,
            "output": ground_truths[i],
            "generated_text": generated_text,
            "user_id": data[i].get("user_id", ""),
            "id": data[i].get("id", ""),
        }
        results.append(rec)

    # ------------------------------
    # Save outputs
    # ------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    results_path_jsonl = os.path.join(args.out_dir, f"{args.wandb_name}_results.jsonl")
    results_path_json = os.path.join(args.out_dir, f"{args.wandb_name}_results.json")

    with open(results_path_jsonl, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(results_path_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved: {results_path_jsonl}")
    print(f"Saved: {results_path_json}")


if __name__ == "__main__":
    main()
