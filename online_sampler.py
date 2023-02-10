from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline
import os

from tqdm import tqdm
import sys
import json
import torch
import argparse
import numpy as np
from utils import MODEL_TYPE, BIG_BENCH_DIR, CACHE_DIR

os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR

parser = argparse.ArgumentParser(
    description="Finetune a transformers model on a causal language modeling task"
)
parser.add_argument(
    "--task",
    type=str,
    default=None,
    help="Which task to use.",
)
parser.add_argument(
    "--task_dir",
    type=str,
    default=None,
    help="Which task to use.",
)
parser.add_argument(
    "--model_dir",
    type=str,
    default=None,
    help="Which model to use.",
)
parser.add_argument(
    "--use_original_model",
    action="store_true",
    help="If passed, use the original model of flan otherwise new model.",
)
parser.add_argument(
    "--sample_size",
    type=int,
    default=4,
    help="Batch size (per device) for the training dataloader.",
)
parser.add_argument(
    "--seed",
    type=int,
    default=1,
    help="seed for the entire algorithm.",
)
# parser.add_argument("--local_rank", type=int, default=0)
args = parser.parse_args()
# Define a task for big-bench
task_name = args.task
device = "cuda:0"
big_bench_dir = BIG_BENCH_DIR
# task_dir = 'tracking_shuffled_objects/three_objects/'
task_dir = args.task_dir + "/"
f = open(big_bench_dir + task_dir + "task.json")
task_data = json.load(f)
if "task_prefix" in task_data.keys():
    task_prefix = task_data["task_prefix"]
else:
    task_prefix = ""
if "example_input_prefix" in task_data.keys():
    example_input_prefix = task_data["example_input_prefix"]
else:
    example_input_prefix = ""
entries = []
targets = []
for (idx, d) in enumerate(task_data["examples"]):
    prefix = task_prefix + example_input_prefix
    if "target_scores" in d.keys():
        options = list(d["target_scores"].keys())
        options = [
            "(" + chr(65 + i) + ") " + option for i, option in enumerate(options)
        ]
        options = "\n".join(options)
        entry = (
            prefix.replace("\n\n", "\n")
            + d["input"].replace("\n\n", "")
            + "\nOptions:\n"
            + options
        )
        # entry = d['input'].replace('\n\n', '') + '\nOptions:\n' + options
        # entry = d['input'].replace('\n\n', '\n')
        entries.append(entry)
        # targets.append(list(d['target_scores'].keys())[list(d['target_scores'].values()).index(1)])
        targets.append("(" + chr(65 + list(d["target_scores"].values()).index(1)) + ")")
    else:
        entry = prefix.replace("\n\n", "\n") + d["input"].replace("\n\n", "\n")
        entries.append(entry)
        if isinstance(d["target"], list):
            targets.append(d["target"][1])
        else:
            targets.append(d["target"])
# Random permute dataset
import numpy as np

np.random.seed(args.seed)
index = np.random.permutation(len(task_data["examples"]))
entries = [entries[_index] for _index in index]
targets = [targets[_index] for _index in index]

portion = int(0.8 * len(task_data["examples"]))
entries = entries[:portion]
targets = targets[:portion]

with open(task_name + ".txt", "r") as file:
    cot_prompt = file.read()

context_length = 256
if args.use_original_model:
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_TYPE, torch_dtype=torch.bfloat16
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_TYPE, torch_dtype=torch.bfloat16, device_map="balanced_low_0"
    )
else:
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_TYPE, torch_dtype=torch.bfloat16
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16, device_map="balanced_low_0"
    )

num_params = sum([np.prod(p.size()) for p in model.parameters()])
print("Number of parameters: ", "{:.2e}".format(num_params))

total_outputs = []
all_inputs = []
all_logits = []
all_labels = []
# num_devices = torch.cuda.device_count()
batch_size = 1
num_samples = args.sample_size
correct = 0
for iters in tqdm(range(int(len(entries) / batch_size))):
    # for entry, target in zip(entries, targets):
    entry = entries[iters * batch_size : (iters + 1) * batch_size]
    target = targets[iters * batch_size : (iters + 1) * batch_size]
    prompt = [cot_prompt + "\n" + "Q: " + _entry + "\nA:" for _entry in entry]
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        do_sample=True,
        max_new_tokens=context_length,
        temperature=1.0,
        num_return_sequences=num_samples,
    )

    outputs_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    # Keep only the unique answers
    outputs_text = list(set(outputs_text))
    num_unique = len(outputs_text)

    total_outputs = total_outputs + outputs_text
    all_prompt = []
    for _prompt in prompt:
        all_prompt = all_prompt + [_prompt for _ in range(num_unique)]
    all_inputs = all_inputs + all_prompt
    all_targets = []
    for _target in target:
        all_targets = all_targets + [_target for _ in range(num_unique)]
    all_labels = all_labels + all_targets
    if outputs_text[0].replace(" ", "").replace(".", "").lower() == str(
        target[0]
    ).lower().replace("\n", " ").replace(" ", "").replace(".", ""):
        # if str(target[0]).lower().replace('\n', ' ').replace(' ', '').replace('.', '') in outputs_text[0].replace(' ', '').replace('.', '').lower():
        correct += 1

data = {}
for i, output in enumerate(total_outputs):
    data.update({i: output})
if args.use_original_model:
    with open(task_name + "_response.json", "w") as outfile:
        for entry_input, entry_output, target in zip(
            all_inputs, total_outputs, all_labels
        ):
            json.dump(
                {"text": entry_input, "labels": entry_output, "target": target}, outfile
            )
            outfile.write("\n")
else:
    with open(task_name + "_response.json", "a") as outfile:
        for entry_input, entry_output, target in zip(
            all_inputs, total_outputs, all_labels
        ):
            json.dump(
                {"text": entry_input, "labels": entry_output, "target": target}, outfile
            )
            outfile.write("\n")
print("Total correct: ", correct)

if args.use_original_model:
    f = open(f"sampler_{task_name}.txt", "w")
else:
    f = open(f"sampler_{task_name}.txt", "a")

f.write(f"correct{correct}\n")
f.close()
