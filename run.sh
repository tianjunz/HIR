# export CUDA_HOME=/usr/local/cuda-11.6/
# export TRANSFORMERS_CACHE=/work/tianjun/few-shot-learning/cache/
## TASK 0
#TASK=logical_deduction_seven_objects
#TASK_DIR=logical_deduction/seven_objects
## TASK 1
TASK=logical_deduction_five_objects
TASK_DIR=logical_deduction/five_objects
## TASK 2
#TASK=logical_deduction_three_objects
#TASK_DIR=logical_deduction/three_objects
#
## TASK 3
#TASK=tracking_shuffled_objects_seven_objects
#TASK_DIR=tracking_shuffled_objects/seven_objects
## TASK 4
#TASK=tracking_shuffled_objects_five_objects
#TASK_DIR=tracking_shuffled_objects/five_objects
## TASK 5
#TASK=tracking_shuffled_objects_three_objects
#TASK_DIR=tracking_shuffled_objects/three_objects
#
#
## TASK 6
#TASK=object_counting
#TASK_DIR=object_counting
## TASK 7
#TASK=date_understanding
#TASK_DIR=date_understanding
## TASK 8
#TASK=penguins_in_a_table
#TASK_DIR=penguins_in_a_table
## TASK 9
#TASK=geometric_shapes
#TASK_DIR=geometric_shapes
## TASK 10
#TASK=reasoning_about_colored_objects
#TASK_DIR=reasoning_about_colored_objects
## TASK 11
#TASK=word_sorting
#TASK_DIR=word_sorting

NUM_SAMPLE=1
NUM_EPOCHS=10
random_port=34515
echo $TASK $TASK_DIR $NUM_SAMPLE $NUM_EPOCHS
cp CoT.txt "$TASK".txt
# rm "$TASK"_response.json;
#torchrun --nproc_per_node 1 --master_port $random_port evaluation.py --use_original_model --task $TASK --task_dir $TASK_DIR --use_cot --model_dir $TASK;
torchrun --nproc_per_node 1 --master_port $random_port online_sampler.py --use_original_model --task $TASK --task_dir $TASK_DIR --sample_size $NUM_SAMPLE --model_dir $TASK;
python -m torch.distributed.launch --nproc_per_node 4 --master_port $random_port --use_env offline_trainer.py --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --output_dir $TASK --use_original_model --task $TASK --task_dir $TASK_DIR --with_tracking --gradient_accumulation_steps 16 --checkpointing_steps last --num_train_epochs $NUM_EPOCHS;
torchrun --nproc_per_node 1 evaluation.py --task $TASK --master_port $random_port --task_dir $TASK_DIR --use_cot --model_dir $TASK;

for i in `seq 1 30`
do
        echo $i
        torchrun --nproc_per_node 1 --master_port $random_port online_sampler.py --task $TASK --task_dir $TASK_DIR --sample_size $NUM_SAMPLE --model_dir $TASK;
        python -m torch.distributed.launch --nproc_per_node 4 --master_port $random_port --use_env offline_trainer.py --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --output_dir $TASK --task $TASK --task_dir $TASK_DIR --with_tracking --gradient_accumulation_steps 16 --checkpointing_steps last --resume_from_checkpoint "$TASK"/last/ --num_train_epochs $NUM_EPOCHS;
        torchrun --nproc_per_node 1 --master_port $random_port evaluation.py --task $TASK --task_dir $TASK_DIR --use_cot --model_dir $TASK;
done
date
