import os
import json
import torch
import datetime
import logging
from tqdm import tqdm
from pathlib import Path
from config import arg_parse
import wandb  # Add wandb import

from gliner import GLiNER
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


def convert_protein_relations_to_edges(protein_relations):
    """
    Convert protein relationship strings like "MT-CO1 -> MT-CO2" to tuple format
    
    Args:
        protein_relations (list): List of strings like "protein1 -> protein2"
        
    Returns:
        list: List of tuples like [(MT-CO1, MT-CO2), (MT-CO1, MT-CO3), ...]
    """
    # Extract all edges as tuples with original protein names
    edges = []
    for relation in protein_relations:
        if "->" in relation:
            source, target = relation.split("->")
            source = source.strip()
            target = target.strip()
            
            # Add edge as tuple
            edges.append((source, target))
    
    return edges

def precision_at_k(predicted_list, ground_truth_set, k):
    top_k_preds = predicted_list[:k]
    correct = sum(1 for p in top_k_preds if p in ground_truth_set)
    return correct / k if k > 0 else 0

def calculate_metrics(predicted_list, ground_truth_list):
    """
    Calculate evaluation metrics comparing predicted proteins with ground truth.
    
    Args:
        predicted_list (list): List of predicted proteins (HGNC symbols)
        ground_truth_list (list): List of ground truth proteins (HGNC symbols)
        
    Returns:
        tuple: A tuple containing (overlap_count, precision, recall, f1_score, jaccard, precision_at_5, precision_at_10)
    """
    # Print original lengths
    original_predicted_len = len(predicted_list)
    original_ground_truth_len = len(ground_truth_list)
    # Convert lists to sets for intersection/union operations
    predicted_set = set(predicted_list)
    # Check if ground truth should be truncated
    if len(ground_truth_list) > len(predicted_list):
        logger.info(f"[NOTE] Ground truth set truncated from {len(ground_truth_list)} to {len(predicted_list)} items")
        ground_truth_set = set(ground_truth_list[:len(predicted_list)])
    else:
        ground_truth_set = set(ground_truth_list)
    # Print final sizes after potential truncation
    logger.info(f"Evaluation using {len(predicted_set)} predicted items vs {len(ground_truth_set)} ground truth items")
    # Calculate basic metrics
    overlap_set = ground_truth_set.intersection(predicted_set)
    overlap_count = len(overlap_set)
    precision = overlap_count / len(predicted_set) if len(predicted_set) > 0 else 0
    # Calculate recall against the original (untruncated) ground truth set for fairness
    original_ground_truth_set = set(ground_truth_list)
    original_overlap_count = len(original_ground_truth_set.intersection(predicted_set))
    recall = original_overlap_count / len(original_ground_truth_set) if original_overlap_count > 0 else 0
    # F1 score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    # Jaccard Similarity against truncated set
    union_count = len(ground_truth_set.union(predicted_set))
    jaccard = overlap_count / union_count if union_count > 0 else 0
    # Precision at K - use original ground truth list for coverage
    precision_at_5 = precision_at_k(predicted_list, original_ground_truth_set, k=5)
    precision_at_10 = precision_at_k(predicted_list, original_ground_truth_set, k=10)
    return overlap_count, precision, recall, f1_score, jaccard, precision_at_5, precision_at_10

def build_and_print_results(initial_reasoning, llm_1st_step_response_hgnc_list, 
                           llm_1st_metrics, crispr_answer):
    """
    Build comprehensive results dictionary and print evaluation metrics
    
    Args:
        initial_reasoning (str): Initial reasoning from LLM first step
        llm_1st_step_response_hgnc_list (list): List of proteins from LLM first step
        llm_1st_metrics (tuple): Metrics for LLM first step (overlap_count, precision, recall, f1_score, jaccard, precision_at_5, precision_at_10)
        crispr_answer (list): Ground truth protein list
        
    Returns:
        dict: Comprehensive results dictionary with all inputs, outputs and metrics
    """
    # Unpack metrics
    llm_1st_overlap_count, llm_1st_precision, llm_1st_recall, llm_1st_f1_score, llm_1st_jaccard, llm_1st_precision_at_5, llm_1st_precision_at_10 = llm_1st_metrics
    # Build comprehensive results dictionary
    results_dict = {
        "inputs": {
            "initial_reasoning": initial_reasoning,
            "llm_1st_step_response_hgnc_list": llm_1st_step_response_hgnc_list
        },
        "evaluation_results": {
            "llm_1st": {
                "precision": llm_1st_precision,
                "recall": llm_1st_recall,
                "f1_score": llm_1st_f1_score,
                "overlap_count": llm_1st_overlap_count,
                "jaccard": llm_1st_jaccard,
                "precision@5": llm_1st_precision_at_5,
                "precision@10": llm_1st_precision_at_10
            }
        }
    }
    # Print evaluation results
    logger.info("\n********** Evaluation Results **********")
    logger.info("\nLLM 1st Step Results:")
    logger.info(f"Precision: {llm_1st_precision:.4f}")
    logger.info(f"Recall: {llm_1st_recall:.4f}")
    logger.info(f"F1 Score: {llm_1st_f1_score:.4f}")
    logger.info(f"Overlap Count: {llm_1st_overlap_count}/{len(crispr_answer)}")
    logger.info(f"Jaccard Similarity: {llm_1st_jaccard:.4f}")
    logger.info(f"Precision at 5: {llm_1st_precision_at_5:.4f}")
    logger.info(f"Precision at 10: {llm_1st_precision_at_10:.4f}")
    return results_dict

def save_results_to_json(sample_id, results_dict, output_dir, timestamp=None):
    """
    Saves the processing results to a JSON file, appending new sample results
    to an existing file rather than creating multiple files.
    
    Args:
        sample_id (str): The sample ID used as the key in the JSON file
        results_dict (dict): Dictionary containing all the result variables
        output_dir (str): Directory to save the results file
        timestamp (str, optional): Timestamp to use for consistent file naming across a run
    
    Returns:
        str: Path to the saved file
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    # Generate timestamp for the filename if not provided
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Create a single filename for the entire run
    filename = f"ft_llm_results_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    # Check if the file already exists
    if os.path.exists(filepath):
        # Load existing data
        with open(filepath, "r") as f:
            try:
                all_results = json.load(f)
            except json.JSONDecodeError:
                # File exists but is empty or corrupted
                logger.warning(f"Warning: Existing file {filepath} appears to be corrupted. Creating new file.")
                all_results = {}
    else:
        # Start with empty dict for new file
        all_results = {}
    # Add new entry for this sample
    all_results[sample_id] = results_dict
    
    # Save entire updated dictionary back to file
    with open(filepath, "w") as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"Results for sample {sample_id} appended to {filepath}")
    return filepath

# Run function
def run_llm(llm_pipeline, prompt, max_new_tokens=1000):
    response = llm_pipeline(prompt, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7)
    return response[0]["generated_text"].replace(prompt, "").strip()

# Step 1: Initial CoT protein reasoning with multi-omics context
def generate_protein_prompt(sample_data):
    cell_line_name = sample_data["cell_line_name"]
    disease = sample_data["disease"]

    top_gene = sample_data["input"]["top_k_gene"]["hgnc_symbols"]
    top_transcript = sample_data["input"]["top_k_transcript"]["hgnc_symbols"]
    top_protein = sample_data["input"]["top_k_protein"]["hgnc_symbols"]
    disease_protein = sample_data["input"]["knowledge_graph"]["disease_protein"]["hgnc_symbols"]
    protein_kg_relation = sample_data["input"]["knowledge_graph"]["protein_relationships"]

    # Truncate top_gene/transcript/protein lists for input compactness with k items
    k = 10
    if len(top_gene) > k:
        top_gene = top_gene[:k]
    if len(top_transcript) > k:
        top_transcript = top_transcript[:k]
    if len(top_protein) > k:
        top_protein = top_protein[:k]

    # Truncate long lists for input compactness
    if len(disease_protein) > 50:
        disease_protein = disease_protein[:50]
        disease_protein.append("... (Only a subset of disease related proteins shown; full list truncated for brevity.)")

    # Option 1: Original format
    # Truncate protein relationships if too long (limit to 100 items)
    if len(protein_kg_relation) > 100:
        protein_kg_relation = protein_kg_relation[:100]
        protein_kg_relation.append("... (Only a subset of protein-protein interactions shown; full list truncated for brevity.)")

    # # Option 2: Talk like a graph
    # # Convert to edge format (list of tuples)
    # protein_kg_relation_edges = convert_protein_relations_to_edges(protein_kg_relation)
    # # Add a string note about truncation to display later
    # protein_kg_relation = [f"({src}, {tgt})" for src, tgt in protein_kg_relation_edges]
    # # Add truncation note if necessary
    # if len(protein_kg_relation) > 100:
    #     protein_kg_relation = protein_kg_relation[:100]
    #     protein_kg_relation.append("... (Only a subset of protein-protein interactions shown; full list truncated for brevity.)")

    return (
        f"[Instruction] Identify the 100 priority genes whose knockout causes the strongest negative effect "
        f"on the viability or proliferation of the {cell_line_name} cell line in the context of {disease}, "
        f"based on the highest relevance values derived from multi-omics datasets and knowledge graph information.\n"
        f"[Input]\n"
        f"- Top 10 ranked genes with copy number values due to strong amplification: {', '.join(top_gene)}\n"
        f"- Top 10 ranked transcripts from transcriptomic profiling with high expression: {', '.join(top_transcript)}\n"
        f"- Top 10 ranked proteins from RPPA proteomics with high expression or activation: {', '.join(top_protein)}\n"
        f"- Disease-relevant proteins extracted from the biomedical knowledge graph: {', '.join(disease_protein)}\n"
        f"- Known protein-protein and disease-protein interactions from the knowledge graph: {', '.join(protein_kg_relation)}\n"
        f"[Output]"
    )

# Generate the finetuning dataset
def prepare_training_data(qa_info_data):
    """
    Prepare training data for fine-tuning from the QA dataset
    
    Args:
        qa_info_data (dict): Dictionary containing QA information
        
    Returns:
        list: List of formatted prompt-response pairs for training
    """
    training_data = []
    for sample_id, sample_info in qa_info_data.items():
        # Generate prompt
        prompt = generate_protein_prompt(sample_info)
        # Get ground truth answer
        ground_truth = sample_info["ground_truth_answer"]["top_bm_gene"]["hgnc_symbols"]
        # Construct response
        response = (
            f"Based on the integrated multi-omics data and knowledge graph, I identified the 100 genes whose knockout "
            f"is predicted to have the most severe negative impact on the viability or proliferation of the "
            f"{sample_info['cell_line_name']} cell line in {sample_info['disease']}.\n\n"
            f"The prioritized gene list is as follows:\n\n"
        )
        for i, gene in enumerate(ground_truth[:100]):
            response += f"{i+1}. {gene}\n"
        response += "\nThese genes represent critical vulnerabilities for the given cell line under the disease context."

        training_example = {
            "text": f"<s>[INST] {prompt.strip()} [/INST] {response.strip()}</s>"
        }
        training_data.append(training_example)
    logger.info(f"Prepared {len(training_data)} training examples")
    return training_data

class WandbLossCallback(TrainerCallback):
    """Custom callback to log loss at each step to wandb"""
    
    def __init__(self):
        super().__init__()
        # Get local rank from environment variable
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.is_main_process = self.local_rank == 0
    
    def on_step_end(self, args, state, control, logs=None, **kwargs):
        """Log step loss to wandb"""
        if not self.is_main_process:
            return
            
        if logs is not None and "loss" in logs:
            step = state.global_step
            wandb.log({"train/loss": logs["loss"]}, step=step)
            # Also log learning rate at each step
            if "learning_rate" in logs:
                wandb.log({"train/learning_rate": logs["learning_rate"]}, step=step)
            
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Log evaluation metrics to wandb"""
        if not self.is_main_process:
            return
            
        if metrics is not None:
            wandb.log({"eval/loss": metrics.get("eval_loss")}, step=state.global_step)

def finetune(args, device, run_timestamp):
    logger.info(f"Starting LLM finetuning run with timestamp: {run_timestamp}")
    
    # Initialize wandb only on the main process
    wandb_project_name = "GALAX"
    wandb_run_name = f"TargetQA-{run_timestamp}"
    
    # Get local rank from environment variable set by DeepSpeed/distributed training
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main_process = local_rank == 0
    
    if is_main_process:
        logger.info("Initializing wandb on main process")
        wandb.init(
            project=wandb_project_name,
            name=wandb_run_name,
            config={
                "base_model": './checkpoints/BioEntity-LLM-20250421_045108/checkpoint-336',
                "num_epochs": 5,
                "batch_size": 1,
                "learning_rate": 1e-5,
                "gradient_accumulation_steps": 2,
                "warmup_ratio": 0.1,
                "weight_decay": 0.0
            }
        )
    else:
        # Disable wandb on non-main processes
        logger.info(f"Process rank {local_rank} - wandb disabled")
        os.environ["WANDB_MODE"] = "disabled"
        # Also initialize wandb in disabled mode to prevent issues
        wandb.init(mode="disabled")
    
    # Check if a local model path is provided
    base_model_name = './checkpoints/BioEntity-LLM-20250421_045108/checkpoint-336'
    logger.info(f"Loading model from local path: {base_model_name}")

    finetuned_model_name = wandb_run_name
    output_dir = os.path.join("checkpoints", finetuned_model_name)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    json_path = "./data/TargetQA/target_qa_k10_bm100_tr.json"
    logger.info(f"Loading QA data from {json_path}")
    with open(json_path, "r") as f:
        qa_info_data = json.load(f)

    training_data = prepare_training_data(qa_info_data)
    
    # Shuffle data before splitting to ensure random distribution
    import random
    random.seed(42)
    random.shuffle(training_data)
    logger.info("Data shuffled before train/val split")
    
    # Then split the data
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
        num_train_epochs=5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        save_steps=335,
        save_strategy="steps",
        eval_steps=335,
        logging_steps=1,  # Log every step
        learning_rate=1e-5,
        weight_decay=0.0,
        fp16=False,
        bf16=True,
        max_grad_norm=0.5,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        report_to="wandb",
        save_total_limit=5,
        gradient_checkpointing=True,
        optim="adamw_torch",
        remove_unused_columns=False,
        deepspeed="./ds_z3_config.json",
        # Enable data shuffling parameters that actually exist in TrainingArguments
        seed=42,
        dataloader_pin_memory=True,
        dataloader_drop_last=False,
        dataloader_num_workers=4,
        # Disable length batching to maintain full randomness
        group_by_length=False
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
    
    # Initialize the Trainer with DeepSpeed config and our custom callback
    wandb_callback = WandbLossCallback()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[wandb_callback]  # Add the wandb callback
    )

    logger.info("Starting full model fine-tuning process...")
    train_result = trainer.train()
    logger.info(f"Training completed. Results: {train_result}")
    
    # Log final metrics to wandb
    wandb.log({"final_loss": train_result.training_loss})
    
    logger.info(f"Saving finetuned model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(f"# TargetQA Fine-tuned Model\n\n")
        f.write(f"Base model: {base_model_name}\n")
        f.write(f"Fine-tuned on: {run_timestamp}\n")
        f.write(f"Training examples: {len(training_data)}\n")
        f.write(f"Fine-tuning method: Full model fine-tuning with DeepSpeed\n")
        f.write(f"Description: Model fine-tuned for protein identification in CRISPR screening data\n")

    # Finish wandb run
    wandb.finish()

    logger.info(f"Finetuning complete. Model saved to {output_dir}")
    return output_dir


# Test prompts from test JSON file
def load_test_prompts_from_json(json_path, num_samples=2):
    """
    Load test prompts from the first N samples in the test JSON file
    
    Args:
        json_path (str): Path to the test JSON file
        num_samples (int): Number of samples to load (default: 2)
        
    Returns:
        list: List of formatted test prompts
    """
    import json
    
    logger.info(f"Loading test prompts from {json_path}")
    with open(json_path, "r") as f:
        test_data = json.load(f)
    
    test_prompts = []
    # Get first num_samples keys from the dictionary
    sample_ids = list(test_data.keys())[:num_samples]
    
    for sample_id in sample_ids:
        sample_data = test_data[sample_id]
        # Use the same generate_protein_prompt function for consistency
        prompt = generate_protein_prompt(sample_data)
        test_prompts.append(prompt)
        
    logger.info(f"Generated {len(test_prompts)} test prompts from test data")
    return test_prompts


def load_and_test_model(model_path, device):
    """
    Load a fine-tuned model and test it with sample prompts
    
    Args:
        model_path (str): Path to the fine-tuned model
        device (str): Device to load the model on ('cuda' or 'cpu')
        
    Returns:
        tuple: A tuple containing (model, tokenizer, pipe)
    """
    logger.info(f"Loading fine-tuned model from {model_path}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    logger.info(f"Model loaded successfully. Creating pipeline...")
    
    # Create a text generation pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )
    
    # Test prompts
    test_json_path = "./data/TargetQA/target_qa_k10_bm100_te.json"
    test_prompts = load_test_prompts_from_json(test_json_path, num_samples=2)
    
    logger.info("Running test prompts...")
    for i, prompt in enumerate(test_prompts):
        logger.info(f"\n\n--- Test Prompt {i+1} ---")
        logger.info(f"Prompt: {prompt}...\n")
        
        # Generate text
        start_time = datetime.datetime.now()
        response = run_llm(pipe, prompt, max_new_tokens=2000)
        end_time = datetime.datetime.now()
        
        generation_time = (end_time - start_time).total_seconds()
        logger.info(f"Generation time: {generation_time:.2f} seconds")
        logger.info(f"Response:\n{response}\n")
        
        # Also print to console for immediate viewing
        print(f"Response:\n{response}\n")
    
    logger.info("Model testing completed")
    return model, tokenizer, pipe


if __name__ == "__main__":
    args = arg_parse()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Generate a single timestamp for the entire run
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Configure wandb - set your API key if needed
    wandb.login(key="fd3ec6d2acb46f89bfff7990f841291f15325b18")  # Uncomment if you need to login manually
    
    # Check device - optimize for H100
    if torch.cuda.is_available():
        device = 'cuda:0'
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"Using GPU: {gpu_name}")
        if 'H100' in gpu_name:
            logger.info("H100 GPU detected - using optimized settings")
        else:
            logger.info(f"Note: Expected H100 GPU, but found {gpu_name}")
    else:
        device = 'cpu'
        logger.warning("No GPU detected, using CPU (not recommended for finetuning)")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Run finetuning with the generated timestamp
    finetune(args, device, run_timestamp)

    # # Model path to load
    # model_path = './checkpoints/TargetQA-20250424_164126/checkpoint-75'
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