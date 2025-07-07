import os
import json
import torch
import datetime
import logging
from tqdm import tqdm
from pathlib import Path
from config import arg_parse
import wandb

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    AutoConfig,
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
# Import flash attention modules
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, LlamaAttention

# Import flash_attn if available
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
    FLASH_ATTENTION_AVAILABLE = True
    print("✅ FlashAttention-2 is available")
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False
    print("⚠️ FlashAttention-2 is not available. Please install it using: pip install flash-attn")


# FlashAttention implementation for Llama models
class LlamaFlashAttention(LlamaAttention):
    """
    Llama attention module using FlashAttention-2 for faster training
    """
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx)
        self.use_flash_attn = FLASH_ATTENTION_AVAILABLE
        
        # Fix for attribute names - check what attributes are available
        if hasattr(self, "num_attention_heads"):
            self.num_heads = self.num_attention_heads
        else:
            # Fallback to config if the attribute is stored there
            self.num_heads = config.num_attention_heads
            
        # Similar approach for num_key_value_heads
        if hasattr(self, "num_key_value_heads"):
            self.num_key_value_heads = self.num_key_value_heads
        elif hasattr(config, "num_key_value_heads"):
            self.num_key_value_heads = config.num_key_value_heads
        else:
            self.num_key_value_heads = self.num_heads
            
        # Get hidden size from config and set it as an attribute
        if hasattr(self, "hidden_size"):
            pass  # Already exists
        else:
            self.hidden_size = config.hidden_size
            
        # Get head dimension
        self.head_dim = self.hidden_size // self.num_heads

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,  # Add this parameter
        position_embeddings=None,  # Add this new parameter for Llama-3
        num_items_in_batch=None,  # Add this new parameter
        **kwargs  # Add this to catch any other unexpected arguments
    ):
        if not self.use_flash_attn:
            return super().forward(
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                output_attentions,
                use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,  # Pass position_embeddings to parent
                num_items_in_batch=num_items_in_batch,
                **kwargs
            )
        
        bsz, q_len, _ = hidden_states.size()
        
        # Get query, key, value projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape query, key, value for multihead attention
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings - with position_embeddings support
        if position_embeddings is not None:
            # Use the provided position embeddings directly
            query_states = query_states * position_embeddings
            key_states = key_states * position_embeddings
        elif position_ids is not None:
            # Use RoPE as before
            cos, sin = self.rotary_emb(value_states, seq_len=q_len)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        
        # Cache key, value states for incremental decoding
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
            
        past_key_value = (key_states, value_states) if use_cache else None

        # Run FlashAttention
        if attention_mask is None:
            # Standard case with no attention mask
            attn_output = flash_attn_func(
                query_states.transpose(1, 2),  # [bs, q_len, n_heads, head_dim]
                key_states.transpose(1, 2),     # [bs, kv_len, n_heads, head_dim]
                value_states.transpose(1, 2),   # [bs, kv_len, n_heads, head_dim]
                dropout_p=0.0,
                softmax_scale=1.0 / math.sqrt(self.head_dim),
                causal=True
            )
            attn_output = attn_output.transpose(1, 2)  # [bs, n_heads, q_len, head_dim]
        else:
            # Handle attention mask (more complex case)
            # Convert attention mask to proper format for FlashAttention
            attention_mask = attention_mask.squeeze(1).squeeze(1)  # [bs, seq_len]
            
            # Get indices of non-masked tokens
            indices = torch.nonzero(attention_mask, as_tuple=True)
            max_seq_len = attention_mask.sum(dim=-1).max().item()
            
            # Unpad input based on attention mask
            q_unpad, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(
                query_states.transpose(1, 2), attention_mask
            )
            k_unpad, indices_k, cu_seqlens_k, max_seqlen_k = unpad_input(
                key_states.transpose(1, 2), attention_mask
            )
            v_unpad, indices_v, cu_seqlens_v, max_seqlen_v = unpad_input(
                value_states.transpose(1, 2), attention_mask
            )
            
            # Run FlashAttention with variable sequence lengths
            output_unpad = flash_attn_varlen_func(
                q_unpad,
                k_unpad,
                v_unpad,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                dropout_p=0.0,
                softmax_scale=1.0 / math.sqrt(self.head_dim),
                causal=True
            )
            
            # Pad output back to original shape
            attn_output = pad_input(
                output_unpad, indices_q, bsz, q_len
            ).reshape(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            
        # Final projection
        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, None, past_key_value


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


# Apply FlashAttention to a model
def apply_flash_attention(model):
    if not FLASH_ATTENTION_AVAILABLE:
        logger.warning(" FlashAttention-2 is not available. Using standard attention instead.")
        return model
    
    logger.info("Applying FlashAttention-2 to the model...")
    # Replace all attention modules with flash attention
    for idx, layer in enumerate(model.model.layers):
        layer.self_attn = LlamaFlashAttention(model.config, layer_idx=idx)
    
    return model

class WandbCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            wandb.log(metrics)
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            wandb.log(logs)

def finetune(args, device):
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"Starting LLM finetuning run with timestamp: {run_timestamp}")
    
    # base_model_name = "meta-llama/Llama-2-7b-chat-hf"
    base_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    finetuned_model_name = f"BioEntity-LLM-{run_timestamp}"
    output_dir = os.path.join("checkpoints", finetuned_model_name)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Initialize wandb here
    wandb.init(
        project="galax",
        name=finetuned_model_name,  # Use finetuned_model_name for the run name
        config={
            "base_model": base_model_name,
            "learning_rate": 1e-5, # You might want to pass these from args or define them here
            "epochs": 3, # Example value, adjust as needed
            "batch_size": 16, # Example value, adjust as needed
            "gradient_accumulation_steps": 8, # Example value, adjust as needed
            "warmup_ratio": 0.1, # Example value, adjust as needed
            "flash_attention": FLASH_ATTENTION_AVAILABLE,
            "run_timestamp": run_timestamp,
            "output_dir": output_dir
        }
    )

    # Set up a file handler to log to the model directory
    log_file_path = os.path.join(output_dir, "training_log.txt")
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info(f"Training logs will be saved to {log_file_path}")

    # You can use either the original QA data or the new jsonl data based on args
    jsonl_path = "./data/TargetPretrain/mixed_description.jsonl"
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
        trust_remote_code=True,
        attn_implementation="flash_attention_2"  # Built-in FlashAttention-2
    )
    
    # Log model configuration
    logger.info(f"Model configuration: {model.config}")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=8,
        save_steps=336,
        save_strategy="steps",
        eval_steps=336,
        logging_steps=1,
        learning_rate=1e-5,
        weight_decay=0.0,
        fp16=False,
        bf16=True,
        max_grad_norm=1.0,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        report_to="wandb",
        save_total_limit=5,
        gradient_checkpointing=True,
        optim="adamw_torch",
        remove_unused_columns=False,
        deepspeed="//ds_z3_config.json"
    )
    
    # Log training arguments
    logger.info(f"Training arguments: {training_args}")

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
        data_collator=data_collator,
        callbacks=[WandbCallback()]
    )
    
    logger.info("Starting full model fine-tuning process with FlashAttention-2...")
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
        f.write(f"Fine-tuning method: Full model fine-tuning with DeepSpeed and FlashAttention-2\n")
        f.write(f"Description: Model fine-tuned for protein identification in CRISPR screening data\n")

    logger.info(f"✅ Finetuning complete. Model saved to {output_dir}")
    
    # Log model info to wandb
    wandb.config.update({
        "training_examples": len(training_data),
        "model_output_dir": output_dir,
        "timestamp": run_timestamp
    })
    
    return output_dir


def load_and_test_model(model_path, device):
    """
    Load a fine-tuned model and test it with sample prompts
    
    Args:
        model_path (str): Path to the fine-tuned model
        device (str): Device to load the model on ('cuda' or 'cpu')
    """
    logger.info(f"Loading fine-tuned model from {model_path}")
    finetuned_model_name

    wandb.init(
        project="galax",
        name=f"test-{Path(model_path).parent.name}-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}", # Adjusted name for testing
        config={"model_path": model_path}
    )
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    # First load the config to ensure compatibility
    model_config = AutoConfig.from_pretrained(model_path)
    # Then load the model with the same config
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=model_config,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    logger.info(f"Model loaded successfully. Creating pipeline...")
    
    # Create a text generation pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )
    
    # Test prompts
    test_prompts = [
        "[Instruction] Identify 10 key signaling proteins involved in breast cancer.\n[Input]\n- Top-ranked genes (gene-level): ESR1, ERBB2, TP53, PIK3CA, BRCA1, BRCA2, EGFR, MYC, PTEN, CDH1\n[Reasoning]",
        
        "[Instruction] Explain the role of TP53 in cancer signaling pathways.\n[Input]\n- Provide a comprehensive explanation of TP53's function in regulating cell cycle, apoptosis, and how its mutations contribute to cancer progression.\n[Reasoning]"
    ]
    
    logger.info("Running test prompts...")
    for i, prompt in enumerate(test_prompts):
        logger.info(f"\n\n--- Test Prompt {i+1} ---")
        logger.info(f"Prompt: {prompt[:100]}...\n")

        wandb.log({
            f"prompt_{i+1}": prompt,
            f"response_{i+1}": response,
            f"generation_time_{i+1}": generation_time
        })
        
        # Generate text
        start_time = datetime.datetime.now()
        response = run_llm(pipe, prompt, max_new_tokens=300)
        end_time = datetime.datetime.now()
        
        generation_time = (end_time - start_time).total_seconds()
        logger.info(f"Generation time: {generation_time:.2f} seconds")
        logger.info(f"Response:\n{response}\n")
        
        # Also print to console for immediate viewing
        print(f"\n--- Test Prompt {i+1} ---")
        print(f"Response:\n{response}\n")
    
    logger.info("Model testing completed")
    return model, tokenizer, pipe


if __name__ == "__main__":
    # Add import needed for the LlamaFlashAttention implementation
    import math
    
    args = arg_parse()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Check if flash-attention is installed
    if not FLASH_ATTENTION_AVAILABLE:
        logger.warning("FlashAttention-2 is not installed. To install it, run: pip install flash-attn")
        response = input("Do you want to continue without FlashAttention-2? (y/n): ")
        if response.lower() != 'y':
            logger.info("Exiting. Please install flash-attn package first.")
            exit()

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

    # # Model path to load
    # model_path = './Checkpoints/finetuned_model/BioEntity-LLM-20250421_045108/checkpoint-336'
    # # Load and test the model
    # model, tokenizer, pipe = load_and_test_model(model_path, device)
    # # Interactive mode
    # print("\n\nEntering interactive mode. Type 'exit' to quit.")
    # while True:
    #     user_input = input("\nEnter a prompt: ")
    #     if user_input.lower() == 'exit':
    #         break
    #     response = run_llm(pipe, user_input, max_new_tokens=2000)
    #     print(f"\nResponse:\n{response}")