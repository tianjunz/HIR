# The Wisdom of Hindsight Makes Language Models Better Instruction Followers

## Intro

This is an implementation of paper: 

<a href="https://arxiv.org/pdf/2302.05206.pdf">The Wisdom of Hindsight Makes Language Models
Better Instruction Followers</a>

## Citation
If you use this code in your own work, please cite our paper:
```
@article{zhang2023wisdom,
  title={The Wisdom of Hindsight Makes Language Models Better Instruction Followers},
  author={Zhang, Tianjun and Liu, Fangchen and Wong, Justin and Abbeel, Pieter and Gonzalez, Joseph E},
  journal={arXiv preprint arXiv:2302.05206},
  year={2023}
}
```

## Installation
Install BigBench
```
# When creating a new task, replace this with your forked repository (see below)
git clone https://github.com/google/BIG-bench.git
cd BIG-bench
python setup.py sdist
pip install -e .
```
Modify ```BIG_BENCH_DIR``` in ```utils.py``` to be the installation path of BigBench.
```
# Install other dependencies
conda env create -f conda_env.yml
conda activate hir
```

## Train FLAN-T5 on BigBench Tasks
Modify ```MODEL_TYPE``` in ```utils.py``` to be the desired model (e.g. ```google/flan-t5-xl```).

Change ```TASK``` to be the desired BigBench Task (e.g. ```logical_deduction_five_objects```). Then get the results through iterative sampling and training:
```
bash run.sh
```
