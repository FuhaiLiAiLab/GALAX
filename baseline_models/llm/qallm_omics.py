import os
import json
import copy
import torch
import traceback
import datetime
import numpy as np
import pandas as pd
import networkx as nx

from pathlib import Path
from config import arg_parse

from motasg_explainer import (
    build_pretrain_model, 
    build_model, 
    kg_data, 
    pre_embed,
    hgnc_to_bmgc_pt_id,
    bmgc_pt_id_to_hgnc,
    create_candidate_explainer,
    generate_best_graph,
    convert_protein_relations_to_edges,
    convert_edges_to_hgnc_symbols,
    convert_to_original_index,
    visualize_hgnc_edges
)

from gliner import GLiNER
from transformers import BioGptTokenizer, BioGptForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

import requests
import json
from collections import defaultdict

# Check device and select GPU with most available memory
def get_gpu_with_max_free_memory():
    """
    Returns the GPU device ID with the most available memory.
    Returns 'cpu' if no CUDA devices are available.
    """
    if not torch.cuda.is_available():
        return 'cpu'
    
    try:
        # Try to use nvidia-smi to get memory info
        import subprocess
        import re
        
        # Run nvidia-smi to get memory usage info
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'], 
                                        encoding='utf-8')
        # Parse the output to get free memory values
        free_memory = [int(x) for x in result.strip().split('\n')]
        
        # Get the GPU with maximum free memory
        max_free_device_id = free_memory.index(max(free_memory))
        print(f"Selected GPU {max_free_device_id} with {max(free_memory)} MB free memory")
        return f'cuda:{max_free_device_id}'
    
    except (subprocess.CalledProcessError, FileNotFoundError, ImportError) as e:
        # If nvidia-smi fails, try to use torch's built-in function (less accurate)
        print(f"Warning: nvidia-smi failed ({e}). Using first available GPU.")
        try:
            # Get device count and find one with most memory
            device_count = torch.cuda.device_count()
            if device_count == 0:
                return 'cpu'
            elif device_count == 1:
                return 'cuda:0'
            
            # Try to find the GPU with most free memory
            max_free = 0
            max_device = 0
            for device_id in range(device_count):
                torch.cuda.set_device(device_id)
                torch.cuda.empty_cache()
                free_mem = torch.cuda.memory_reserved(device_id) - torch.cuda.memory_allocated(device_id)
                if free_mem > max_free:
                    max_free = free_mem
                    max_device = device_id
            
            print(f"Selected GPU {max_device} based on torch memory stats")
            return f'cuda:{max_device}'
        except:
            # Fall back to the first GPU if all else fails
            return 'cuda:0'


def extract_proteins_with_chatgpt(text, api_key, api_url="https://api.openai.com/v1/chat/completions"):
    """
    Extract protein/gene mentions from text using ChatGPT API.
    
    Args:
        text (str): Text to extract proteins from
        
    Returns:
        list: List of extracted protein/gene names
    """
    # Prepare the prompt for ChatGPT
    prompt = f"""
    Extract all human protein or gene names mentioned in the following text. 
    Return only the gene/protein symbols in a Python list format.
    
    Text: {text}
    """
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        # "model": "gpt-3.5-turbo",
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a biomedical NER system that extracts human protein and gene names."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0,
        "max_tokens": 2000
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response_json = response.json()
        
        if "choices" in response_json and len(response_json["choices"]) > 0:
            protein_text = response_json["choices"][0]["message"]["content"].strip()
            
            # Try to extract the list from the response
            import re
            match = re.search(r'\[(.*?)\]', protein_text, re.DOTALL)
            if match:
                items = match.group(1).split(',')
                proteins = [item.strip().strip("'\"") for item in items if item.strip()]
                return proteins
            else:
                # Fallback: extract words that look like gene symbols
                return re.findall(r'\b[A-Z0-9]+\b', protein_text)
        else:
            print(f"Error in API response: {response_json}")
            return []
    except Exception as e:
        print(f"Error calling ChatGPT API: {e}")
        return []

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
    # Print lengths
    predicted_len = len(predicted_list)
    ground_truth_len = len(ground_truth_list)
    # Convert lists to sets for intersection/union operations
    predicted_set = set(predicted_list)
    ground_truth_set = set(ground_truth_list)
    # Print final sizes after potential truncation
    print(f"Evaluation using {len(predicted_set)} predicted items vs {len(ground_truth_set)} ground truth items")
    # Calculate basic metrics
    overlap_set = ground_truth_set.intersection(predicted_set)
    overlap_count = len(overlap_set)
    precision = overlap_count / len(predicted_set) if len(predicted_set) > 0 else 0
    # Calculate recall against the ground truth set for fairness
    recall = overlap_count / len(ground_truth_set) if overlap_count > 0 else 0
    # F1 score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    # Jaccard Similarity against truncated set
    union_count = len(ground_truth_set.union(predicted_set))
    jaccard = overlap_count / union_count if union_count > 0 else 0
    # Precision at K - use ground truth list for coverage
    precision_at_5 = precision_at_k(predicted_list, ground_truth_set, k=5)
    precision_at_10 = precision_at_k(predicted_list, ground_truth_set, k=10)
    return overlap_count, precision, recall, f1_score, jaccard, precision_at_5, precision_at_10

def build_and_print_results(initial_reasoning, llm_1st_step_response_hgnc_list,
                            llm_1st_metrics, crispr_answer, crispr_answer_dict):
    """
    Build comprehensive results dictionary and print evaluation metrics
    
    Args:
        initial_reasoning (str): Initial reasoning from LLM
        llm_1st_step_response_hgnc_list (list): List of proteins from LLM 1st step
        llm_1st_metrics (tuple): Metrics from LLM 1st step
        crispr_answer (list): List of ground truth proteins
        crispr_answer_dict (dict): Dictionary of ground truth proteins
        
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
        },
        "crispr_answer": crispr_answer_dict
    }

    # Print evaluation results
    print("\n********** Evaluation Results **********")
    print("\nLLM 1st Step Results:")
    print(f"Precision: {llm_1st_precision:.4f}")
    print(f"Recall: {llm_1st_recall:.4f}")
    print(f"F1 Score: {llm_1st_f1_score:.4f}")
    print(f"Overlap Count: {llm_1st_overlap_count}/{len(crispr_answer)}")
    print(f"Jaccard Similarity: {llm_1st_jaccard:.4f}")
    print(f"Precision at 5: {llm_1st_precision_at_5:.4f}")
    print(f"Precision at 10: {llm_1st_precision_at_10:.4f}")
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
    filename = f"qallm_omics_results_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    # Check if the file already exists
    if os.path.exists(filepath):
        # Load existing data
        with open(filepath, "r") as f:
            try:
                all_results = json.load(f)
            except json.JSONDecodeError:
                # File exists but is empty or corrupted
                print(f"Warning: Existing file {filepath} appears to be corrupted. Creating new file.")
                all_results = {}
    else:
        # Start with empty dict for new file
        all_results = {}
    # Add new entry for this sample
    all_results[sample_id] = results_dict
    # Save entire updated dictionary back to file
    with open(filepath, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results for sample {sample_id} appended to {filepath}")
    return filepath

# Run function
def run_llm(llm_pipeline, prompt, max_new_tokens=4000):
    response = llm_pipeline(prompt, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7)
    return response[0]["generated_text"].replace(prompt, "").strip()

# Step 1: Initial CoT gene/protein reasoning
def generate_proteins_prompt(sample_data):
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

    return (
            f"[Instruction] Identify the 100 priority genes whose knockout causes the strongest negative effect "
            f"on the viability or proliferation of the {cell_line_name} cell line in the context of {disease}, "
            f"based on the highest relevance values derived from multi-omics datasets and knowledge graph information.\n"
            f"[Input]\n"
            f"- Top 10 ranked genes with copy number values due to strong amplification: {', '.join(top_gene)}\n"
            f"- Top 10 ranked transcripts from transcriptomic profiling with high expression: {', '.join(top_transcript)}\n"
            f"- Top 10 ranked proteins from RPPA proteomics with high expression or activation: {', '.join(top_protein)}\n"
            f"[Output]"
        )


def train(args, device):
    # Path to the saved file
    json_path = "./QA_Data/multi_sample_qa_info_k100_bm100_te.json"
    # Load QA info data
    with open(json_path, "r") as f: qa_info_data = json.load(f)
    # Load KG omics data
    xAll_omics, omics_node_index_df, name_embeddings, desc_embeddings, all_edge_index, internal_edge_index, ppi_edge_index, ppi_edges = kg_data(args, device)
    # Load mapping dictionary between ID and Index
    nodeid_index_data = pd.read_csv('./BMG/DTI_data/nodes_index.csv')
    nodeid_index_dict = dict(zip(nodeid_index_data['Node'], nodeid_index_data['Index']))
    index_nodeid_dict = dict(zip(nodeid_index_data['Index'], nodeid_index_data['Node']))
    bmgc_protein_df = pd.read_csv('./BMG/BioMedGraphica-Conn/Entity/Protein/BioMedGraphica_Conn_Protein.csv')
    # number of entity
    args.num_entity = xAll_omics.shape[1]
    num_entity = xAll_omics.shape[1]

    # Load pretrain model
    pretrain_model = build_pretrain_model(args, device)
    pretrain_model.load_state_dict(torch.load(args.save_path))
    pretrain_model.eval()
    # Load model
    downstream_model = build_model(args, device)
    downstream_model.load_state_dict(torch.load(args.train_save_path))
    downstream_model.eval()
    # Load LLM model
    # Option 1: Use general LLM model
    # model_name = "meta-llama/Llama-2-7b-chat-hf"
    # model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    # model_name = './Checkpoints/finetuned_model/BioEntity-LLM-20250421_045108/checkpoint-336'
    model_name = './Checkpoints/finetuned_model/CRISPR-QA-20250425_014540/checkpoint-125'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    llm_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
    llm_model = llm_model.to(device)  # explicitly move to CUDA or CPU
    # # Build pipeline
    llm_pipeline = pipeline("text-generation", model=llm_model, tokenizer=tokenizer)
    # # Option 2: Use BioGPT model
    # # model_name = "microsoft/biogpt"
    # model_name = "microsoft/BioGPT-Large"
    # tokenizer = BioGptTokenizer.from_pretrained(model_name)
    # llm_model = BioGptForCausalLM.from_pretrained(model_name)
    # # Build pipeline
    # llm_pipeline = pipeline('text-generation', model=llm_model, tokenizer=tokenizer)
    ner_model_type = "ChatGPT"
    # Load NER model
    if ner_model_type == "GLiNER":
        gliner_model_name = "gliner-community/gliner_large-v2.5"
        gliner_model = GLiNER.from_pretrained(gliner_model_name)
        labels = ["gene"]
    elif ner_model_type == "ChatGPT":
        # Replace NER model with ChatGPT API
        api_key = "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"  # Replace with actual API key
        api_url = "https://api.openai.com/v1/chat/completions"
    
    # Create a unified timestamp for this run
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Starting Plain LLM run with timestamp: {run_timestamp}")
    # Path to plotted files
    plots_main_dir = os.path.join(".", "Plots")  # Create main Plots directory if it doesn't exist
    if not os.path.exists(plots_main_dir): os.makedirs(plots_main_dir)
    # Create timestamped subfolder within the main Plots directory
    plots_dir = os.path.join(plots_main_dir, f"Run_{run_timestamp}")
    if not os.path.exists(plots_dir): os.makedirs(plots_dir)

    # Training based on data
    for sample_id, sample_info in qa_info_data.items():
        # ***************************************************************************************************************************
        # **************************************************** LLM 1st STEP *********************************************************
        # ***************************************************************************************************************************
        # ************************************** LLM Reasoning with Prompt ************************************
        # import pdb; pdb.set_trace()
        print(f"\n{'='*80}")
        print(f"🧬 Running Plain LLM Model on Sample ID: {sample_id}")
        print(f"{'='*80}\n")
        prompt1 = generate_proteins_prompt(sample_info)
        print(f"\n{'='*50}")
        print("Step 1 - Initial Prompt:\n", prompt1)
        print(f"\n{'='*50}")
        print("\n")
        print("\n")
        initial_reasoning = run_llm(llm_pipeline, prompt1)
        # Truncate the initial reasoning after symbol '</s>' or '</p>'
        if '</s>' in initial_reasoning:
            initial_reasoning = initial_reasoning.split('</s>')[0]
        elif '</p>' in initial_reasoning:
            initial_reasoning = initial_reasoning.split('</p>')[0]
        elif '</INST>' in initial_reasoning:
            initial_reasoning = initial_reasoning.split('</INST>')[0]
        print("Step 1 - Initial LLM Reasoning:\n", initial_reasoning)
        print("\n")
        # ********************************** Convert LLM Protein to Graph Node ********************************
        if ner_model_type == "GLiNER":
            # Use GLiNER model to extract entities
            entities = gliner_model.predict_entities(initial_reasoning, labels)
            # Reduce the duplicated entities
            unique_entities = dict({(entity["text"], entity["label"]) for entity in entities})
            # Convert the unique entities to a list of proteins
            llm_1st_step_response_hgnc_list = list(unique_entities.keys())
            print("\nStep 1 - LLM Gene/Protein List:\n", llm_1st_step_response_hgnc_list)
        elif ner_model_type == "ChatGPT":
            # Use ChatGPT API to extract entities
            llm_1st_step_response_hgnc_list = extract_proteins_with_chatgpt(initial_reasoning, api_key, api_url)
            print("\nStep 1 - LLM Gene/Protein List:\n", llm_1st_step_response_hgnc_list)
        # convert llm_1st_step_response_hgnc_list to bmgc_id list by hgnc_nodeid_dict
        llm_1st_step_bmgc_id_hgnc_dict, llm_1st_step_response_bmgc_id_list = hgnc_to_bmgc_pt_id(llm_1st_step_response_hgnc_list, bmgc_protein_df)
        # convert llm_1st_step_response_bmgc_id_list to llm_protein_index by nodeid_index_dict
        llm_protein_index = [nodeid_index_dict[bmgc_id] for bmgc_id in llm_1st_step_response_bmgc_id_list if bmgc_id in nodeid_index_dict]

        # ***************************************************************************************************************************
        # *********************************************************** RESULTS *******************************************************
        # ***************************************************************************************************************************
        # ************************************** Evaluate and save results **************************************
        # import pdb; pdb.set_trace()
        # Get ground truth answer
        crispr_answer = sample_info["ground_truth_answer"]["top_bm_gene"]["hgnc_symbols"]
        crispr_answer_dict = sample_info["ground_truth_answer"]
        
        # Calculate evaluation metrics
        llm_1st_metrics = calculate_metrics(llm_1st_step_response_hgnc_list, crispr_answer)
        
        # Build comprehensive results dictionary and print metrics
        results_dict = build_and_print_results(
            initial_reasoning=initial_reasoning,
            llm_1st_step_response_hgnc_list=llm_1st_step_response_hgnc_list,
            llm_1st_metrics=llm_1st_metrics,
            crispr_answer=crispr_answer,
            crispr_answer_dict=crispr_answer_dict
        )
        # Save the results to JSON file
        save_results_to_json(sample_id=sample_id, 
                            results_dict=results_dict,
                            output_dir=args.output_result_dir,
                            timestamp=run_timestamp)
        # # Clear CUDA cache to free up GPU memory
        # torch.cuda.empty_cache()


if __name__ == "__main__":
    args = arg_parse()
    # Check device
    device = 'cpu' if args.device < 0 else (f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Train the model
    train(args, device)