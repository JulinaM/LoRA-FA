import re
import os 
import json 
import math
from fire import Fire
from tqdm import tqdm
from utils import initialize_text_to_text_model, load_peft_model
from human_eval.data import write_jsonl, read_problems


# This template formats the prompts for the chat model
template = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.

### User:
{instruction}

### Assistant:
"""


def main(wandb_name):
    print("wandb_name  -->", wandb_name)
    problems = read_problems()
    if "test" in wandb_name:
        problems = dict(list(problems.items())[:5])
    
    model, tokenizer = initialize_text_to_text_model(
        "meta-llama/Llama-2-7b-hf",
        "CausalLM",
        True,
        flash_attention=True
    )
    model = load_peft_model(model, f'./results/lorafa_wizard_lm/{wandb_name}/9/')
    model = model.to('cuda')
    model.eval()

    # --- Step 2: Load MT-Bench Questions from Local File ---
    question_file_path = "../hf-datasets/mt_bench/question.jsonl"
    print(f"Loading questions from {question_file_path}")

    # Load all questions into a list
    all_questions = []
    with open(question_file_path, "r") as f:
        for line in f:
            all_questions.append(json.loads(line))

    model_answers = []
    model_id = f"{wandb_name}_model"

    # --- Step 3: Generate Model Responses ---
    for question in tqdm(all_questions, desc="Generating Answers"):
        output_turns = []
        
        # Turn 1
        prompt = template.format(instruction=question['turns'][0])
        inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
        output = model.generate(
            **inputs,
            max_new_tokens=1024,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )
        output_text = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        output_turns.append(output_text)
        
        # Turn 2
        context = prompt + output_text + "</s>"
        prompt_2 = context + "### User:\n" + question['turns'][1] + "\n\n### Assistant:\n"
        inputs_2 = tokenizer([prompt_2], return_tensors="pt").to(model.device)
        output_2 = model.generate(
            **inputs_2,
            max_new_tokens=1024,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )
        output_text_2 = tokenizer.decode(output_2[0][inputs_2.input_ids.shape[1]:], skip_special_tokens=True)
        output_turns.append(output_text_2)

        model_answers.append({
            "question_id": question["question_id"],
            "answer_id": f"{model_id}-{question['question_id']}",
            "model_id": model_id,
            "choices": [{"index": 0, "turns": output_turns}],
        })

    # --- Save results ---
    output_file = f"./dialogue_eval/{model_id}_answers.jsonl"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        for answer in model_answers:
            f.write(json.dumps(answer) + "\n")
    print(f"\nModel answers saved to: {output_file}")

if __name__ == "__main__":
    Fire(main)
