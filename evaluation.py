import os
from tqdm import tqdm
import sys
import time
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import json

device = "cuda:0"
from transformers.models.t5.modeling_t5 import T5Block
from transformers import T5Tokenizer, T5ForConditionalGeneration

# import deepspeed
from transformers import pipeline
from utils import BIG_BENCH_DIR, MODEL_TYPE
import argparse

local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "4"))

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
    "--use_original_model",
    action="store_true",
    help="If passed, use the original model of flan otherwise new model.",
)
parser.add_argument(
    "--use_cot",
    action="store_true",
    help="If passed, use the original model of flan otherwise new model.",
)
parser.add_argument(
    "--model_dir",
    type=str,
    default=None,
    help="Which model to use.",
)
parser.add_argument(
    "--seed",
    type=int,
    default=1,
    help="seed for the entire algorithm.",
)
args = parser.parse_args()

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
import numpy as np

num_params = sum([np.prod(p.size()) for p in model.parameters()])
print("Number of parameters: ", "{:.2e}".format(num_params))
task_name = args.task
big_bench_dir = BIG_BENCH_DIR
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
entries = entries[portion:]
targets = targets[portion:]

with open(task_name + ".txt", "r") as file:
    cot_prompt = file.read()

context_length = 256

correct = 0
feedback_correct = 0
inputs = []
for entry in entries:
    example = entry
    if args.use_cot:
        prompt = cot_prompt + "\n\nQ: " + example + "\nA:"
    else:
        prompt = "Q: " + example + "\nA:"
    inputs.append(prompt)

print("number of inputs: ", len(inputs))
outputs = []
batch_size = 1
start_time = time.time()
for iters in tqdm(range(int(len(inputs) / batch_size))):
    tokenized_inputs = tokenizer(
        inputs[iters * batch_size : (iters + 1) * batch_size], return_tensors="pt"
    ).to(device)
    tokenized_outputs = model.generate(
        **tokenized_inputs, do_sample=False, max_new_tokens=context_length
    )
    responses = tokenizer.batch_decode(tokenized_outputs, skip_special_tokens=True)
    outputs = outputs + responses
end_time = time.time()
print("total time used: ", end_time - start_time)

# Write the OpenAI respones to file
data = {}
for i, output in enumerate(outputs):
    data.update({i: output})
import json

with open(task_name + "_response1.json", "w") as outfile:
    json.dump(data, outfile)

no_answer = 0
# assert len(entries) == len(outputs)
for entry, response, target in zip(entries, outputs, targets):
    round_correct = False
    prediction = response.replace("\n", " ")
    print(prediction)
    if prediction.replace(" ", "").replace(".", "").lower() == str(
        target
    ).lower().replace("\n", " ").replace(" ", "").replace(".", ""):
        correct += 1
print("accuracy is ", correct / len(entries))
print("total correct ", correct)
print("total no answer ", no_answer)

if args.use_original_model:
    f = open(f"evaluation_{task_name}.txt", "w")
else:
    f = open(f"evaluation_{task_name}.txt", "a")

f.write(f"accuracy_{correct/len(entries)}_correct_{correct}_no_answer_{no_answer}\n")
f.close()
