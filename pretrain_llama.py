import os
import json
import torch
import datetime
import logging
from tqdm import tqdm
from pathlib import Path
from config import arg_parse

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    pipeline,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from datasets import Dataset
from sklearn.model_selection import train_test_split

import trl
from trl import SFTTrainer
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from pathlib import Path
import os
import datetime
import json
import torch
import logging
from sklearn.model_selection import train_test_split

from accelerate import Accelerator


# Run function
def run_llm(llm_pipeline, prompt, max_new_tokens=2000):
    response = llm_pipeline(prompt, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7)
    return response[0]["generated_text"].replace(prompt, "").strip()

def prepare_training_data_from_jsonl(jsonl_path):
    """
    Prepare training data from a JSONL file containing entity descriptions
    
    Args:
        jsonl_path (str): Path to the JSONL file with entity descriptions
        
    Returns:
        list: List of formatted text examples for training
    """
    training_data = []
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            example = json.loads(line)
            # The text field contains the entire description
            training_example = {
                "text": example["text"]
            }
            training_data.append(training_example)
    
    logger.info(f"Prepared {len(training_data)} training examples from JSONL")
    return training_data

def finetune(args, device):
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"Starting LLM finetuning run with timestamp: {run_timestamp}")
    
    # base_model_name = "meta-llama/Llama-2-7b-chat-hf"
    base_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    finetuned_model_name = f"BioEntity-LLM-{run_timestamp}"
    output_dir = os.path.join("Checkpoints", "finetuned_model", finetuned_model_name)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # You can use either the original QA data or the new jsonl data based on args
    jsonl_path = "./QA_Data/mixed_description.jsonl"
    logger.info(f"Loading entity description data from {jsonl_path}")
    training_data = prepare_training_data_from_jsonl(jsonl_path)

        
    train_data, val_data = train_test_split(training_data, test_size=0.1, random_state=42)
    logger.info(f"Split data into {len(train_data)} training and {len(val_data)} validation examples")

    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(val_data)
    
    logger.info(f"Loading tokenizer for model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading model for full fine-tuning: {base_model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        save_steps=4004,
        save_strategy="steps",
        eval_steps=4004,
        logging_steps=1,
        learning_rate=1e-5,
        weight_decay=0.0,
        fp16=False,
        bf16=True,
        max_grad_norm=1.0,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        report_to="none",
        save_total_limit=5,
        gradient_checkpointing=True,
        optim="adamw_torch",
        remove_unused_columns=False,
        deepspeed="./ds_z3_config.json"
    )

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=2048)

    logger.info("Preprocessing datasets...")
    tokenized_train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=["text"]
    )
    tokenized_eval_dataset = eval_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=["text"]
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Initialize the Trainer with DeepSpeed config
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    
    logger.info("Starting full model fine-tuning process...")
    train_result = trainer.train()
    logger.info(f"✅ Training completed. Results: {train_result}")

    logger.info(f"Saving finetuned model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(f"# CRISPR-QA Fine-tuned Model\n\n")
        f.write(f"Base model: {base_model_name}\n")
        f.write(f"Fine-tuned on: {run_timestamp}\n")
        f.write(f"Training examples: {len(training_data)}\n")
        f.write(f"Fine-tuning method: Full model fine-tuning with DeepSpeed\n")
        f.write(f"Description: Model fine-tuned for protein identification in CRISPR screening data\n")

    logger.info(f"✅Finetuning complete. Model saved to {output_dir}")
    return output_dir


if __name__ == "__main__":
    args = arg_parse()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Check device - optimize for H100
    if torch.cuda.is_available():
        device = 'cuda:0'  # Use the first GPU, which should be the H100
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"Using GPU: {gpu_name}")
        if 'H100' in gpu_name:
            logger.info("H100 GPU detected - using optimized settings")
        else:
            logger.info(f"Note: Expected H100 GPU, but found {gpu_name}")
    else:
        device = 'cpu'
        logger.warning("No GPU detected, using CPU (not recommended for finetuning)")
    
    # Run finetuning
    finetune(args, device)

    # # Load and eval the finetuned model
    # model_path = "Checkpoints/finetuned_model/CRISPR-QA-20231005_123456"  # Replace with your model path