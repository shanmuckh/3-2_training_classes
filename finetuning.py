# ===============================
# FLAN-T5 + LoRA Fine-tuning
# Dataset: knkarthick/samsum
# ===============================

import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["DISABLE_TF_IMPORTS"] = "1"

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType

# ===============================
# CONFIGURATION
# ===============================
MODEL_NAME = "google/flan-t5-base"
DATASET_NAME = "knkarthick/samsum"
OUTPUT_DIR = "./flan_t5_lora"

EPOCHS = 3
BATCH_SIZE = 4
LEARNING_RATE = 2e-4
MAX_INPUT_LEN = 512
MAX_TARGET_LEN = 128

# ===============================
# LOAD DATASET
# ===============================
print("📥 Loading SAMSum dataset from Hugging Face...")
dataset = load_dataset(DATASET_NAME)

print(dataset)

# ===============================
# LOAD MODEL & TOKENIZER
# ===============================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# ===============================
# APPLY LoRA
# ===============================
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q", "v"]
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ===============================
# PREPROCESSING
# ===============================
def preprocess(batch):
    inputs = [
        "Summarize the following dialogue:\n" + dialogue
        for dialogue in batch["dialogue"]
    ]

    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LEN,
        truncation=True,
        padding="max_length"
    )

    labels = tokenizer(
        batch["summary"],
        max_length=MAX_TARGET_LEN,
        truncation=True,
        padding="max_length"
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(
    preprocess,
    batched=True,
    remove_columns=dataset["train"].column_names
)

# ===============================
# TRAINING ARGUMENTS
# ===============================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    num_train_epochs=EPOCHS,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    fp16=torch.cuda.is_available(),
    report_to="none"
)

# ===============================
# TRAINER
# ===============================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer
)

# ===============================
# TRAIN
# ===============================
trainer.train()

# ===============================
# SAVE LoRA ADAPTER
# ===============================
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("🎉 Training completed successfully")
print("📦 LoRA adapter saved at:", OUTPUT_DIR)
