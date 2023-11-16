export MODEL_DIR="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="output"

export DATASET_NAME="face_synthetics.py"
export TRAIN_DATA_DIR="./dataset_100"

accelerate launch train.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --seed=42 \
 --dataset_name=$DATASET_NAME \
 --train_data_dir=$TRAIN_DATA_DIR \
 --resolution=512 \
 --learning_rate=1e-5 \
 --validation_image "./dataset_100/000000_cond.png" "./dataset_100/000099_cond.png" \
 --validation_prompt "a man in a gray shirt standing in a field" "a man in a suit and a hat" \
 --train_batch_size=4 \
 --gradient_accumulation_steps=1 \
 --gradient_checkpointing \
 --use_8bit_adam \
 --mixed_precision="bf16" \
 --num_train_epochs=100 \
 --validation_steps=100 \