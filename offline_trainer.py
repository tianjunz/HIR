#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
on a text file or a dataset without using HuggingFace Trainer.
Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import argparse
import json
import logging
import math
import os
import random
from itertools import chain
from pathlib import Path
from torch import nn

import datasets
from datasets import DatasetDict
import torch
from datasets import concatenate_datasets, load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from huggingface_hub import Repository
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers.utils import (
    check_min_version,
    get_full_repo_name,
    send_example_telemetry,
)
from transformers.utils.versions import require_version
from verifier import *
from transformers.trainer_pt_utils import LabelSmoother
from torch.distributions import Categorical
from transformers import DataCollatorForLanguageModeling, DataCollatorForSeq2Seq
from datasets import set_caching_enabled
from utils import BIG_BENCH_DIR, MODEL_TYPE
import numpy as np

datasets.disable_caching()

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.24.0")

logger = get_logger(__name__)

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a causal language modeling task"
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=8,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="constant",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to store the final model."
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--no_keep_linebreaks",
        action="store_true",
        help="Do not keep line breaks when using TXT files.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--hub_token", type=str, help="The token to use to push to the Model Hub."
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"` and `"comet_ml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    # Added parser
    parser.add_argument(
        "--use_original_model",
        action="store_true",
        help="If passed, use the original model of flan otherwise new model.",
    )
    parser.add_argument(
        "--kl_loss",
        action="store_true",
        help="If passed, use the kl loss otherwise no kl loss.",
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

    args = parser.parse_args()
    if args.push_to_hub:
        assert (
            args.output_dir is not None
        ), "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


def main():
    args = parse_args()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm_no_trainer", args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["logging_dir"] = args.output_dir

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        **accelerator_log_kwargs,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(
                    Path(args.output_dir).name, token=args.hub_token
                )
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    task_name = args.task
    label_length = 16
    context_length = 256 - label_length
    # Load dataset
    big_bench_dir = BIG_BENCH_DIR
    task_dir = args.task_dir + "/"
    f = open(big_bench_dir + task_dir + "task.json")
    task_data = json.load(f)
    entries = []
    targets = []
    for (idx, d) in enumerate(task_data["examples"]):
        prefix = ""
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
            entries.append(entry)
            targets.append(
                "(" + chr(65 + list(d["target_scores"].values()).index(1)) + ")"
            )
        else:
            entry = prefix.replace("\n\n", "\n") + d["input"].replace("\n\n", "\n")
            entries.append(entry)
            if isinstance(d["target"], list):
                targets.append(d["target"][1])
            else:
                targets.append(d["target"])

    # Random permute dataset
    np.random.seed(args.seed)
    index = np.random.permutation(len(task_data["examples"]))
    entries = [entries[_index] for _index in index]
    targets = [targets[_index] for _index in index]

    dataset_size = int(0.8 * len(task_data["examples"]))
    val_dataset_size = int(0.2 * len(task_data["examples"]))

    raw_datasets = load_dataset("json", data_files=task_name + "_response.json")
    train_raw_datasets = raw_datasets["train"].train_test_split(
        shuffle=True, test_size=0.1
    )
    train_raw_datasets["train"] = train_raw_datasets["train"].filter(
        lambda example, idx: idx < dataset_size * 4, with_indices=True
    )
    train_raw_datasets["test"] = train_raw_datasets["test"].filter(
        lambda example, idx: idx < val_dataset_size * 4, with_indices=True
    )
    # TODO: sample
    raw_datasets = DatasetDict(
        {
            "train": train_raw_datasets["train"],
            "validation": train_raw_datasets["test"],
        }
    )
    print(raw_datasets)
    if args.use_original_model:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_TYPE)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_TYPE, torch_dtype=torch.bfloat16
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_TYPE)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.output_dir, torch_dtype=torch.bfloat16
        )

    class DynamicLengthSampler:
        def __init__(self, min_value, max_value):
            self.values = list(range(min_value, max_value))

        def __call__(self, max_value):
            if max_value <= 3:
                return 0, max_value
            values = list(range(0, max_value - 3, 2))
            start_idx = np.random.choice(values)
            values = list(range(start_idx + 1, max_value + 1, 2))
            end_idx = np.random.choice(values)
            return start_idx, end_idx

    output_size = DynamicLengthSampler(4, 64)
    total_correct = 0
    total = 0

    def hindsight_relabel(example):
        prefix = example["text"][
            example["text"].rindex("Q:") : example["text"].rindex("A:")
        ]
        (
            hindsight_label,
            neg_hindsight_label,
            correct,
        ) = tracking_shuffled_objects_three_objects(
            prefix, example["labels"], example["target"]
        )
        prefix_index = example["text"].index("\n\n")
        example["text_hind"] = hindsight_label + example["text"][prefix_index:]
        example["neg_text_hind"] = neg_hindsight_label + example["text"][prefix_index:]
        example["text_hind"] = example["text_hind"].lstrip(" ")
        example["neg_text_hind"] = example["neg_text_hind"].lstrip(" ")
        example["text_nohind"] = example["text"][prefix_index:].lstrip("\n")
        example["correct"] = correct
        return example

    raw_datasets = raw_datasets.map(hindsight_relabel)
    for data in raw_datasets["train"]:
        total_correct = total_correct + data["correct"]
        total = total + 1
    print("total ", total, "total correct ", total_correct)
    correct_port = total_correct / total
    incorrect_port = 1 - correct_port

    data_collator = DataCollatorForSeq2Seq(model=model, tokenizer=tokenizer)
    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print("Number of parameters: ", "{:.2e}".format(num_params))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names

    def tokenize_function(examples):
        # TODO: added for now, remove for the future
        outputs = tokenizer(examples["labels"])["input_ids"]
        labels_with_ignore_index = []
        for labels_example in outputs:
            labels_example = [label if label != 0 else -100 for label in labels_example]
            # labels_example = [label for label in labels_example]
            labels_with_ignore_index.append(labels_example)
        targets = tokenizer(examples["target"])["input_ids"]
        targets_with_ignore_index = []
        for targets_example in targets:
            targets_example = [
                target if target != 0 else -100 for target in targets_example
            ]
            # labels_example = [label for label in labels_example]
            targets_with_ignore_index.append(targets_example)

        inputs = tokenizer(
            examples["text"],
        )
        inputs_hind = tokenizer(
            examples["text_hind"],
        )
        neg_inputs_hind = tokenizer(
            examples["neg_text_hind"],
        )

        inputs_nohind = tokenizer(
            examples["text_nohind"],
        )
        return {
            "input_ids": inputs_hind["input_ids"],
            "attention_mask": inputs_hind["attention_mask"],
            "input_ids_nohind": inputs_nohind["input_ids"],
            "input_ids_orig": inputs["input_ids"],
            "attention_mask_orig": inputs["attention_mask"],
            "attention_mask_nohind": inputs_nohind["attention_mask"],
            "input_ids_neg": neg_inputs_hind["input_ids"],
            "attention_mask_neg": neg_inputs_hind["attention_mask"],
            "labels": labels_with_ignore_index,
            "targets": targets_with_ignore_index,
            "correct": examples["correct"],
        }

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )
    print(tokenized_datasets)
    lm_datasets = tokenized_datasets
    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    (
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )
    # model_ref = accelerator.prepare(model_ref)
    # value = accelerator.prepare(value)

    # Prepare label smoother
    epsilon = 0.2
    label_smoother = LabelSmoother(epsilon=epsilon)

    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    if accelerator.distributed_type == DistributedType.TPU:
        model.tie_weights()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config[
            "lr_scheduler_type"
        ].value
        accelerator.init_trackers("clm_no_trainer", experiment_config)

    # Train!
    total_batch_size = (
        args.per_device_train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Num Processes = {accelerator.num_processes}")
    print(
        "batch size ",
        total_batch_size,
        args.per_device_train_batch_size,
        accelerator.num_processes,
        args.gradient_accumulation_steps,
    )
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[
                -1
            ]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

    # update the progress_bar if load from checkpoint
    progress_bar.update(starting_epoch * num_update_steps_per_epoch)
    completed_steps = starting_epoch * num_update_steps_per_epoch

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        # model_ref.eval()
        if args.with_tracking:
            total_loss = 0
        for step, batch in enumerate(train_dataloader):
            # We need to skip steps until we reach the resumed step
            with accelerator.accumulate(model):
                start_length, length = output_size(batch["labels"].shape[1])
                truncate = length < batch["labels"].shape[1]
                if truncate and tokenizer.decode(
                    batch["labels"][0, :length], skip_special_tokens=True
                ).replace(" ", "").replace(".", "").lower() == tokenizer.decode(
                    batch["targets"][0], skip_special_tokens=True
                ).lower().replace(
                    "\n", " "
                ).replace(
                    " ", ""
                ).replace(
                    ".", ""
                ):
                    if batch["correct"][0] == 0:
                        batch["input_ids"], batch["input_ids_neg"] = (
                            batch["input_ids_neg"],
                            batch["input_ids"],
                        )
                    batch["input_ids"] = torch.cat(
                        [
                            batch["input_ids"][:, :-1],
                            batch["labels"][:, :start_length],
                            batch["input_ids"][:, -1:],
                        ],
                        dim=1,
                    )
                    batch["input_ids_neg"] = torch.cat(
                        [
                            batch["input_ids_neg"][:, :-1],
                            batch["labels"][:, :start_length],
                            batch["input_ids_neg"][:, -1:],
                        ],
                        dim=1,
                    )
                    batch["labels"] = batch["labels"][:, start_length:length]

                    batch["labels"] = tokenizer(
                        [
                            tokenizer.decode(
                                batch["labels"][0], skip_special_tokens=True
                            )
                        ],
                        return_tensors="pt",
                    )["input_ids"].to(batch["labels"].device)
                elif truncate and tokenizer.decode(
                    batch["labels"][0, :length], skip_special_tokens=True
                ).replace(" ", "").replace(".", "").lower() in tokenizer.decode(
                    batch["targets"][0], skip_special_tokens=True
                ).lower().replace(
                    "\n", " "
                ).replace(
                    " ", ""
                ).replace(
                    ".", ""
                ):
                    if batch["correct"][0] == 0:
                        batch["input_ids"], batch["input_ids_neg"] = (
                            batch["input_ids_neg"],
                            batch["input_ids"],
                        )
                    batch["input_ids"] = torch.cat(
                        [
                            batch["input_ids"][:, :-1],
                            batch["labels"][:, :start_length],
                            batch["input_ids"][:, -1:],
                        ],
                        dim=1,
                    )
                    batch["input_ids_neg"] = torch.cat(
                        [
                            batch["input_ids_neg"][:, :-1],
                            batch["labels"][:, :start_length],
                            batch["input_ids_neg"][:, -1:],
                        ],
                        dim=1,
                    )
                    batch["labels"] = batch["labels"][:, start_length:length]
                else:
                    # batch['labels'] = tokenizer([tokenizer.decode(batch['labels'][0, :length], skip_special_tokens=True)], return_tensors="pt")['input_ids'].to(batch['labels'].device)
                    batch["input_ids"] = torch.cat(
                        [
                            batch["input_ids"][:, :-1],
                            batch["labels"][:, :start_length],
                            batch["input_ids"][:, -1:],
                        ],
                        dim=1,
                    )
                    batch["input_ids_neg"] = torch.cat(
                        [
                            batch["input_ids_neg"][:, :-1],
                            batch["labels"][:, :start_length],
                            batch["input_ids_neg"][:, -1:],
                        ],
                        dim=1,
                    )
                    batch["labels"] = batch["labels"][:, start_length:length]
                    batch["labels"] = tokenizer(
                        [
                            tokenizer.decode(
                                batch["labels"][0], skip_special_tokens=True
                            )
                        ],
                        return_tensors="pt",
                    )["input_ids"].to(batch["labels"].device)
                outputs = model(batch["input_ids"], labels=batch["labels"])
                loss = label_smoother(outputs, batch["labels"])
                # TODO: Add label smoother
                bs, seq_len, dim = outputs.logits.shape
                outputs_pos = model(batch["input_ids"], labels=batch["labels"])
                outputs_neg = model(batch["input_ids_neg"], labels=batch["labels"])
                pos_prob = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")(
                    outputs_pos.logits.reshape(-1, dim), batch["labels"].reshape(-1)
                )
                neg_prob = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")(
                    outputs_neg.logits.reshape(-1, dim), batch["labels"].reshape(-1)
                )
                pos_prob = (-pos_prob.reshape(bs, seq_len).mean(-1)).exp()
                neg_prob = (-neg_prob.reshape(bs, seq_len).mean(-1)).exp()
                pos_log_prob = -torch.log(pos_prob / (pos_prob + neg_prob) + 1e-8)
                neg_log_prob = -torch.log(neg_prob / (pos_prob + neg_prob) + 1e-8)
                loss = loss + (1 - epsilon) * pos_log_prob + epsilon * neg_log_prob
                if args.kl_loss:
                    outputs_pos = model(batch["input_ids_orig"], labels=batch["labels"])
                    pred_dist = F.log_softmax(outputs_pos.logits, dim=-1)
                    with torch.no_grad():
                        outputs_ref = model_ref(
                            batch["input_ids_orig"], labels=batch["labels"]
                        )
                        true_dist = F.softmax(outputs_ref.logits, dim=-1)
                    kl_loss = nn.KLDivLoss(reduction="batchmean")(pred_dist, true_dist)
                    loss = loss + 0.001 * kl_loss

                entropy = Categorical(logits=outputs.logits).entropy().mean()
                loss = loss - 0.001 * (entropy)

                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps }"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)
            if completed_steps >= args.max_train_steps:
                break

        model.eval()
        losses = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(batch["input_ids"], labels=batch["labels"])

            loss = outputs.loss
            losses.append(
                accelerator.gather_for_metrics(
                    loss.repeat(args.per_device_eval_batch_size)
                )
            )

        losses = torch.cat(losses)
        try:
            eval_loss = torch.mean(losses)
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")

        logger.info(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}")

        if args.with_tracking:
            accelerator.log(
                {
                    "perplexity": perplexity,
                    "eval_loss": eval_loss,
                    "train_loss": total_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )
        if accelerator.is_main_process:
            print(
                "loss item ",
                eval_loss.item(),
                total_loss.item() / len(train_dataloader),
                "\n",
            )
            print("portion ", correct_port, incorrect_port, "\n")

        if args.push_to_hub and epoch < args.num_train_epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                args.output_dir,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)
                repo.push_to_hub(
                    commit_message=f"Training in progress epoch {epoch}",
                    blocking=False,
                    auto_lfs_prune=True,
                )

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

    if args.checkpointing_steps == "last":
        epoch = epoch + 1
        # output_dir = f"epoch_{epoch}"
        output_dir = "last"
        if args.output_dir is not None:
            output_dir = os.path.join(args.output_dir, output_dir)
        accelerator.save_state(output_dir)

    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
        )
        if accelerator.is_main_process:
            print("main process")
            tokenizer.save_pretrained(args.output_dir)
            if args.push_to_hub:
                repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)

        with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
            json.dump({"perplexity": perplexity}, f)


if __name__ == "__main__":
    main()
