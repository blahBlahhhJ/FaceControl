export MODEL_DIR="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="output"

export DATASET_NAME="face_synthetics.py"
export TRAIN_DATA_DIR="./dataset_100000"

accelerate launch train.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --seed=42 \
 --dataset_name=$DATASET_NAME \
 --train_data_dir=$TRAIN_DATA_DIR \
 --resolution=512 \
 --learning_rate=1e-5 \
 --validation_image "./evaluation_samples/099998_cond.png" "./evaluation_samples/099999_cond.png" \
 --validation_prompt "High-quality close-up photo of a girl standing in a classroom" "Portrait of a man in a suit, looking down" \
 --train_batch_size=4 \
 --gradient_accumulation_steps=1 \
 --gradient_checkpointing \
 --use_8bit_adam \
 --mixed_precision="bf16" \
 --num_train_epochs=1 \
 --validation_steps=2000 \