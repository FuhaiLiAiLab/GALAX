import os
import re
import json
import copy
import torch
import time
import traceback
import datetime
import numpy as np
import pandas as pd
import networkx as nx

from pathlib import Path
from config import arg_parse
from utils import select_best_gpu_device

from motasg_explainer import (
    load_combined_model,
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
                            completed_best_sampled_graph_bmgc_id_list,
                            completed_best_sampled_graph_edge_index,
                            completed_best_sampled_graph_additional_edge_index,
                            completed_best_sampled_graph_hgnc_list, 
                            best_conn_completed_graph_bmgc_id_list,
                            best_conn_completed_graph_edge_index,
                            best_conn_completed_graph_additional_edge_index,
                            best_conn_completed_graph_hgnc_list, 
                            edge_text_descriptions,
                            conn_edge_text_descriptions,
                            hgnc_edges, hgnc_additional_edges,
                            original_edge_index, original_additional_edge_index,
                            conn_hgnc_edges, conn_hgnc_additional_edges,
                            conn_original_edge_index, conn_original_additional_edge_index,
                            refined_reasoning, llm_2nd_step_response_hgnc_list,
                            llm_1st_metrics, kg_metrics, llm_2nd_metrics, crispr_answer, crispr_answer_dict):
    """
    Build comprehensive results dictionary and print evaluation metrics
    
    Args:
        initial_reasoning (str): Initial reasoning from LLM
        llm_1st_step_response_hgnc_list (list): List of proteins from LLM 1st step
        completed_best_sampled_graph_bmgc_id_list (list): List of BMGC IDs for the completed graph
        completed_best_sampled_graph_edge_index (torch.Tensor): Edge index for the completed graph
        completed_best_sampled_graph_additional_edge_index (torch.Tensor): Additional edge index for the completed graph
        completed_best_sampled_graph_hgnc_list (list): List of HGNC symbols for the completed graph
        best_conn_completed_graph_bmgc_id_list (list): List of BMGC IDs for the best connected graph
        best_conn_completed_graph_edge_index (torch.Tensor): Edge index for the best connected graph
        best_conn_completed_graph_additional_edge_index (torch.Tensor): Additional edge index for the best connected graph
        best_conn_completed_graph_hgnc_list (list): List of HGNC symbols for the best connected graph
        edge_text_descriptions (list): List of edge descriptions
        conn_edge_text_descriptions (list): List of connected edge descriptions
        hgnc_edges (list): List of edges in HGNC format
        hgnc_additional_edges (list): List of additional edges in HGNC format
        original_edge_index (torch.Tensor): Original edge index
        original_additional_edge_index (torch.Tensor): Original additional edge index
        conn_hgnc_edges (list): List of connected HGNC edges
        conn_hgnc_additional_edges (list): List of connected additional HGNC edges
        conn_original_edge_index (torch.Tensor): Connected original edge index
        conn_original_additional_edge_index (torch.Tensor): Connected original additional edge index
        refined_reasoning (str): Refined reasoning from LLM
        llm_2nd_step_response_hgnc_list (list): List of proteins from LLM 2nd step
        llm_1st_metrics (tuple): Metrics from LLM 1st step
        kg_metrics (tuple): Metrics from KG
        llm_2nd_metrics (tuple): Metrics from LLM 2nd step
        crispr_answer (list): List of ground truth proteins
        crispr_answer_dict (dict): Dictionary of ground truth proteins
        
    Returns:
        dict: Comprehensive results dictionary with all inputs, outputs and metrics
    """
    # Unpack metrics
    llm_1st_overlap_count, llm_1st_precision, llm_1st_recall, llm_1st_f1_score, llm_1st_jaccard, llm_1st_precision_at_5, llm_1st_precision_at_10 = llm_1st_metrics
    kg_overlap_count, kg_precision, kg_recall, kg_f1_score, kg_jaccard, kg_precision_at_5, kg_precision_at_10 = kg_metrics
    llm_2nd_overlap_count, llm_2nd_precision, llm_2nd_recall, llm_2nd_f1_score, llm_2nd_jaccard, llm_2nd_precision_at_5, llm_2nd_precision_at_10 = llm_2nd_metrics
    
    # Build comprehensive results dictionary
    results_dict = {
        "inputs": {
            "initial_reasoning": initial_reasoning,
            "llm_1st_step_response_hgnc_list": llm_1st_step_response_hgnc_list
        },
        "graph_data": {
            "completed_best_sampled_graph_bmgc_id_list": completed_best_sampled_graph_bmgc_id_list,
            "completed_best_sampled_graph_edge_index": completed_best_sampled_graph_edge_index.tolist(),
            "completed_best_sampled_graph_additional_edge_index": completed_best_sampled_graph_additional_edge_index.tolist(),
            "completed_best_sampled_graph_hgnc_list": completed_best_sampled_graph_hgnc_list,
            "best_conn_completed_graph_bmgc_id_list": best_conn_completed_graph_bmgc_id_list,
            "best_conn_completed_graph_edge_index": best_conn_completed_graph_edge_index.tolist(),
            "best_conn_completed_graph_additional_edge_index": best_conn_completed_graph_additional_edge_index.tolist(),
            "best_conn_completed_graph_hgnc_list": best_conn_completed_graph_hgnc_list,
            "hgnc_edges": hgnc_edges,
            "hgnc_additional_edges": hgnc_additional_edges,
            "original_edge_index": original_edge_index.tolist(),
            "original_additional_edge_index": original_additional_edge_index.tolist(),
            "conn_hgnc_edges": conn_hgnc_edges,
            "conn_hgnc_additional_edges": conn_hgnc_additional_edges,
            "conn_original_edge_index": conn_original_edge_index.tolist(),
            "conn_original_additional_edge_index": conn_original_additional_edge_index.tolist(),
            "edge_text_descriptions": edge_text_descriptions,
            "conn_edge_text_descriptions": conn_edge_text_descriptions
        },
        "outputs": {
            "refined_reasoning": refined_reasoning,
            "llm_2nd_step_response_hgnc_list": llm_2nd_step_response_hgnc_list
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
            },
            "kg": {
                "precision": kg_precision,
                "recall": kg_recall,
                "f1_score": kg_f1_score,
                "overlap_count": kg_overlap_count,
                "jaccard": kg_jaccard,
                "precision@5": kg_precision_at_5,
                "precision@10": kg_precision_at_10
            },
            "llm_2nd": {
                "precision": llm_2nd_precision,
                "recall": llm_2nd_recall,
                "f1_score": llm_2nd_f1_score,
                "overlap_count": llm_2nd_overlap_count,
                "jaccard": llm_2nd_jaccard,
                "precision@5": llm_2nd_precision_at_5,
                "precision@10": llm_2nd_precision_at_10
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

    print("\nKnowledge Graph Results:")
    print(f"Precision: {kg_precision:.4f}")
    print(f"Recall: {kg_recall:.4f}")
    print(f"F1 Score: {kg_f1_score:.4f}")
    print(f"Overlap Count: {kg_overlap_count}/{len(crispr_answer)}")
    print(f"Jaccard Similarity: {kg_jaccard:.4f}")
    print(f"Precision at 5: {kg_precision_at_5:.4f}")
    print(f"Precision at 10: {kg_precision_at_10:.4f}")

    print("\nLLM 2nd Step Results:")
    print(f"Precision: {llm_2nd_precision:.4f}")
    print(f"Recall: {llm_2nd_recall:.4f}")
    print(f"F1 Score: {llm_2nd_f1_score:.4f}")
    print(f"Overlap Count: {llm_2nd_overlap_count}/{len(crispr_answer)}")
    print(f"Jaccard Similarity: {llm_2nd_jaccard:.4f}")
    print(f"Precision at 5: {llm_2nd_precision_at_5:.4f}")
    print(f"Precision at 10: {llm_2nd_precision_at_10:.4f}")
    
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
    filename = f"galax_results_{timestamp}.json"
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

# Step 2: Refined reasoning with context from Step 1 + Graph Generation module
def refine_with_graph_context_prompt(sample_data, initial_reasoning, prev_proteins, graph_context, graph_proteins):
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
        f"[Instruction] Based on initial LLM reasoning and subsignnaling gene regulatory network identified by subgraph generator, "
        f"please identify the 100 priority genes whose knockout causes the strongest negative effect "
        f"on the viability or proliferation of the {cell_line_name} cell line in the context of {disease}."
        f"[Input]\n"
        f"- Top 10 ranked genes with copy number values due to strong amplification: {', '.join(top_gene)}\n"
        f"- Top 10 ranked transcripts from transcriptomic profiling with high expression: {', '.join(top_transcript)}\n"
        f"- Top 10 ranked proteins from RPPA proteomics with high expression or activation: {', '.join(top_protein)}\n"
        f"- Disease-relevant proteins extracted from the biomedical knowledge graph: {', '.join(disease_protein)}\n"
        f"- Known protein-protein and disease-protein interactions from the knowledge graph: {', '.join(protein_kg_relation)}\n"
        f"🔍 ** Identified Subsignnaling Gene Regulatory Network from SubGraph Generator **\n"
        f"- Involved genes in the subgraph: {', '.join(graph_proteins)}\n"
        f"- The following signaling path was inferred to represent a likely cascade:\n"
        f"{graph_context}\n\n"
        f"[Refined Reasoning]"
    )


def train(args, device):
    # Path to the saved file
    json_path = "./data/TargetQA/target_qa_k10_bm100.json"
    # Load QA info data
    with open(json_path, "r") as f: qa_info_data = json.load(f)
    
    # Create a unified timestamp for this run
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Check for existing results file to resume from
    # existing_results_file = "./TargetQA_Results/galax_results_20250828_222614.json"
    existing_results_file = None
    processed_samples = set()

    if existing_results_file is not None and os.path.exists(existing_results_file):
        try:
            with open(existing_results_file, "r") as f:
                existing_results = json.load(f)
                processed_samples = set(existing_results.keys())
                print(f"Found existing results file with {len(processed_samples)} processed samples")
                print(f"Processed samples: {sorted(processed_samples)}")
                # Extract timestamp from the filename
                timestamp_match = re.search(r'galax_results_(\d{8}_\d{6})\.json', existing_results_file)
                run_timestamp = timestamp_match.group(1)
                print(f"Extracted timestamp from existing file: {run_timestamp}")
        except (json.JSONDecodeError, FileNotFoundError):
            print("Could not load existing results file, starting fresh")

    print(f"Starting/Resuming GALAX run with timestamp: {run_timestamp}")
    
    # Load KG omics data
    xAll, node_index_df, name_embeddings, desc_embeddings, all_edge_index, internal_edge_index, ppi_edge_index, ppi_edges = kg_data(args, device)
    # Load mapping dictionary between ID and Index
    nodeid_index_data = pd.read_csv('./data/TargetQA/nodes_index.csv')
    nodeid_index_dict = dict(zip(nodeid_index_data['Node'], nodeid_index_data['Index']))
    index_nodeid_dict = dict(zip(nodeid_index_data['Index'], nodeid_index_data['Node']))
    bmgc_protein_df = pd.read_csv('./data/BioMedGraphica-Conn/Entity/Protein/BioMedGraphica_Conn_Protein.csv')
    # number of entity
    args.num_entity = xAll.shape[1]
    num_entity = xAll.shape[1]

    # Load pretrain model and downstream model
    pretrain_model, downstream_model, checkpoint_info = load_combined_model(args, device)
    pretrain_model.eval()
    downstream_model.eval()
    
    # Load LLM model
    model_name = './checkpoints/TargetQA-20250707_051359/checkpoint-335'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
    llm_model = llm_model.to(device)
    llm_pipeline = pipeline("text-generation", model=llm_model, tokenizer=tokenizer)
    
    ner_model_type = "ChatGPT"
    # Load NER model
    if ner_model_type == "GLiNER":
        gliner_model_name = "gliner-community/gliner_large-v2.5"
        gliner_model = GLiNER.from_pretrained(gliner_model_name)
        labels = ["gene"]
    elif ner_model_type == "ChatGPT":
        api_key = "" # fill your OpenAI API key here
        api_url = "https://api.openai.com/v1/chat/completions"
    
    # Path to plotted files
    plots_main_dir = os.path.join(".", "Plots")
    if not os.path.exists(plots_main_dir): os.makedirs(plots_main_dir)
    plots_dir = os.path.join(plots_main_dir, f"Run_{run_timestamp}")
    if not os.path.exists(plots_dir): os.makedirs(plots_dir)

    # Filter out already processed samples
    remaining_samples = {k: v for k, v in qa_info_data.items() if k not in processed_samples}
    total_samples = len(qa_info_data)
    remaining_count = len(remaining_samples)

    print(f"Total samples: {total_samples}")
    print(f"Already processed: {len(processed_samples)}")
    print(f"Remaining to process: {remaining_count}")

    if remaining_count == 0:
        print("All samples have already been processed!")
        return

    # Training based on remaining data
    for idx, (sample_id, sample_info) in enumerate(remaining_samples.items()):
        current_position = len(processed_samples) + idx + 1
        print(f"\n{'='*80}")
        print(f"🧬 Running GALAX Model on Sample ID: {sample_id} ({current_position}/{total_samples})")
        print(f"{'='*80}\n")
        
        # ***************************************************************************************************************************
        # **************************************************** LLM 1st STEP *********************************************************
        # ***************************************************************************************************************************
        # ************************************** LLM Reasoning with Prompt ************************************
        # import pdb; pdb.set_trace()
        print(f"\n{'='*50}")
        prompt1 = generate_proteins_prompt(sample_info)
        print("Step 1 - Initial Prompt:\n", prompt1)
        print(f"\n{'='*50}")
        print("\n")
        print("\n")
        initial_reasoning = run_llm(llm_pipeline, prompt1)
        # Truncate the initial reasoning after symbol "</s>" or "</p>"
        if "</s>" in initial_reasoning:
            initial_reasoning = initial_reasoning.split("</s>")[0]
        elif "</p>" in initial_reasoning:
            initial_reasoning = initial_reasoning.split("</p>")[0]
        elif "</INST>" in initial_reasoning:
            initial_reasoning = initial_reasoning.split("</INST>")[0]
        elif "[/Output]" in initial_reasoning:
            initial_reasoning = initial_reasoning.split("[/Output]")[0]
        elif "[/Instruction]" in initial_reasoning:
            initial_reasoning = initial_reasoning.split("[/Instruction]")[0]
        elif "[/INST]]" in initial_reasoning:
            initial_reasoning = initial_reasoning.split("[/INST]]")[0]
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
            # Use ChatGPT API to extract entities with retry logic
            max_retries = 3
            retry_delay = 2  # seconds
            for attempt in range(max_retries):
                try:
                    print(f"Attempting ChatGPT API call (attempt {attempt + 1}/{max_retries})...")
                    llm_1st_step_response_hgnc_list = extract_proteins_with_chatgpt(initial_reasoning, api_key, api_url)
                    print("\nStep 1 - LLM Gene/Protein List:\n", llm_1st_step_response_hgnc_list)
                    break  # Success, exit retry loop
                except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.RequestException) as e:
                    print(f"ChatGPT API connection error (attempt {attempt + 1}/{max_retries}): {str(e)}")
                    if attempt < max_retries - 1:
                        print(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        print("All ChatGPT API attempts failed due to connection issues.")
                        print("Using empty list as fallback for protein extraction.")
                        llm_1st_step_response_hgnc_list = []
                        break
                except Exception as e:
                    print(f"ChatGPT API call failed with unexpected error (attempt {attempt + 1}/{max_retries}): {str(e)}")
                    if attempt < max_retries - 1:
                        print(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        print("All ChatGPT API attempts failed.")
                        print("Using empty list as fallback for protein extraction.")
                        llm_1st_step_response_hgnc_list = []
                        break
        # convert llm_1st_step_response_hgnc_list to bmgc_id list by hgnc_nodeid_dict
        llm_1st_step_bmgc_id_hgnc_dict, llm_1st_step_response_bmgc_id_list = hgnc_to_bmgc_pt_id(llm_1st_step_response_hgnc_list, bmgc_protein_df)
        # convert llm_1st_step_response_bmgc_id_list to llm_protein_index by nodeid_index_dict
        llm_protein_index = [nodeid_index_dict[bmgc_id] for bmgc_id in llm_1st_step_response_bmgc_id_list if bmgc_id in nodeid_index_dict]

        # ***************************************************************************************************************************
        # **************************************************** Graph Generation *****************************************************
        # ***************************************************************************************************************************
        # *************************************** Prepare sample data *****************************************
        # Extract omics information
        top_gene_bmgc_id = sample_info["input"]["top_k_gene"]["protein_bmgc_ids"]
        top_transcript_bmgc_id = sample_info["input"]["top_k_transcript"]["protein_bmgc_ids"]
        top_protein_bmgc_id = sample_info["input"]["top_k_protein"]["protein_bmgc_ids"]
        # Truncate top_gene/transcript/protein lists for input compactness with k items
        k = 10
        if len(top_gene_bmgc_id) > k:
            top_gene_bmgc_id = top_gene_bmgc_id[:k]
        if len(top_transcript_bmgc_id) > k:
            top_transcript_bmgc_id = top_transcript_bmgc_id[:k]
        if len(top_protein_bmgc_id) > k:
            top_protein_bmgc_id = top_protein_bmgc_id[:k]
        # Convert "non-existed" to empty lists
        if top_gene_bmgc_id == "non-existed" or not isinstance(top_gene_bmgc_id, list):
            print("⚠️ Warning: top_gene_bmgc_id is marked as non-existed, using empty list")
            top_gene_bmgc_id = []
        if top_transcript_bmgc_id == "non-existed" or not isinstance(top_transcript_bmgc_id, list):
            print("⚠️ Warning: top_transcript_bmgc_id is marked as non-existed, using empty list")
            top_transcript_bmgc_id = []
        if top_protein_bmgc_id == "non-existed" or not isinstance(top_protein_bmgc_id, list):
            print("⚠️ Warning: top_protein_bmgc_id is marked as non-existed, using empty list")
            top_protein_bmgc_id = []
        # Convert bmgc id to index by [nodeid_index_dict]
        top_gene_index = [nodeid_index_dict[bmgc_id] for bmgc_id in top_gene_bmgc_id if bmgc_id in nodeid_index_dict]
        top_transcript_index = [nodeid_index_dict[bmgc_id] for bmgc_id in top_transcript_bmgc_id if bmgc_id in nodeid_index_dict]
        top_protein_index = [nodeid_index_dict[bmgc_id] for bmgc_id in top_protein_bmgc_id if bmgc_id in nodeid_index_dict]

        # Extract sample KG information
        selected_sample_index = sample_info["sample_index"]
        disease_protein_bmgc_id = sample_info["input"]["knowledge_graph"]["disease_protein"]["bmgc_ids"]
        disease_protein_index = sample_info["input"]["knowledge_graph"]["disease_protein"]["indices"]
        ppi_nodes_index = sample_info["input"]["knowledge_graph"]["ppi_neighbors"]["indices"]

        # Encode selected_sample_x using the pretrained model and downstream_model
        selected_sample_x = xAll[selected_sample_index, :].reshape(-1, 1)  # selected sample feature
        selected_sample_x_emb = pre_embed(pretrain_model, downstream_model, selected_sample_x, 
                                        name_embeddings, desc_embeddings, 
                                        all_edge_index, internal_edge_index, ppi_edge_index, device)
        # ******************************* Generate candidate set with setting **********************************
        # Build initial node id list and candidate node id list
        initial_node_id_list = []  # Initialize with empty list to avoid the UnboundLocalError
        if disease_protein_bmgc_id is not None and len(disease_protein_bmgc_id) > 0:
            # Truncate the list if len(disease_protein_bmgc_id) > 20
            if len(disease_protein_bmgc_id) > 20:
                initial_node_id_list = copy.deepcopy(disease_protein_bmgc_id[:20])
            else:
                initial_node_id_list = copy.deepcopy(disease_protein_bmgc_id)
        elif llm_1st_step_response_bmgc_id_list is not None and len(llm_1st_step_response_bmgc_id_list) > 0:
            # Truncate the list if len(llm_1st_step_response_bmgc_id_list) > 20
            if len(llm_1st_step_response_bmgc_id_list) > 20:
                initial_node_id_list = copy.deepcopy(llm_1st_step_response_bmgc_id_list[:20])
            else:
                initial_node_id_list = copy.deepcopy(llm_1st_step_response_bmgc_id_list)
        else:
            # Union of all omics protein ids \mathcal{G}_2^{(\text{sub})}
            initial_node_id_list = sorted(list(set(top_gene_bmgc_id + top_transcript_bmgc_id + top_protein_bmgc_id)))
            # randomly select 10 elements if there are more than 10
            if len(initial_node_id_list) > 20:
                initial_node_id_list = np.random.choice(initial_node_id_list, size=20, replace=False).tolist()
                # initial_node_id_list = copy.deepcopy(llm_1st_step_response_bmgc_id_list[:20])

        # Option 1: Candidate node from LLM reasoned proteins

        # Option 2: Candidate node from disease-related proteins
        # Build up the condidate node index
        candidate_node_index = sorted(list(set(disease_protein_index + top_gene_index + top_transcript_index + top_protein_index + ppi_nodes_index + llm_protein_index)))
        candidate_node_id = [index_nodeid_dict[i] for i in candidate_node_index]
        # Extract feature for candidate nodes
        candidate_feat = selected_sample_x_emb[candidate_node_index, :]
        # Create candidate set dictionary
        candidate_set = {}
        for i, protein_id in enumerate(candidate_node_id):
            protein_feature = candidate_feat[i].clone()
            candidate_set[protein_id] = protein_feature
        # ******************************************* Create explainer ******************************************
        # Initialize retry counters and success flag
        max_retries = 6
        retry_count = 0
        graph_generation_successful = False
        # Track parameters for each attempt
        attempt_params = []

        while retry_count < max_retries and not graph_generation_successful:
            try:
                # Log retry attempt
                if retry_count > 0:
                    print(f"\n🔄 Retry attempt {retry_count}/{max_retries-1} for sample {sample_id}")
                # Adjust parameters based on retry count
                current_epochs = max(2, 5 - retry_count)  # Reduce epochs with each retry
                current_lr = 0.001 * (1 + retry_count)  # Increase learning rate slightly with each retry
                current_samples = max(5, 10 - retry_count)  # Reduce num_samples with each retry
                current_nodes = max(100, 200 - retry_count * 25)  # Reduce max nodes with each retry
                current_steps = max(20, 50 - retry_count * 5)  # Reduce max steps with each retry
                # Track parameters used in this attempt
                attempt_params.append({
                    "retry": retry_count,
                    "epochs": current_epochs,
                    "lr": current_lr,
                    "samples": current_samples,
                    "nodes": current_nodes,
                    "steps": current_steps
                })
                
                # Add a validation check before creating the explainer
                if len(initial_node_id_list) < 1 or len(candidate_set) < 2:
                    print(f"⚠️ Warning: Insufficient nodes for graph generation for sample {sample_id}.")
                    print(f"Initial nodes: {len(initial_node_id_list)}, Candidate nodes: {len(candidate_set)}")
                    raise ValueError("Not enough nodes to form a valid graph")
                # On retry, adjust initial node list if possible
                if retry_count > 0:
                    # Alternate between disease proteins and LLM proteins
                    if retry_count % 2 == 1 and disease_protein_bmgc_id is not None and len(disease_protein_bmgc_id) > 0:
                        max_initial = min(20, len(disease_protein_bmgc_id))
                        initial_node_id_list = copy.deepcopy(disease_protein_bmgc_id[:max_initial])
                        print(f"Using disease proteins as initial nodes: {len(initial_node_id_list)} nodes")
                    elif retry_count % 2 == 0 and llm_1st_step_response_bmgc_id_list is not None and len(llm_1st_step_response_bmgc_id_list) > 0:
                        max_initial = min(20, len(llm_1st_step_response_bmgc_id_list))
                        initial_node_id_list = copy.deepcopy(llm_1st_step_response_bmgc_id_list[:max_initial])
                        print(f"Using LLM proteins as initial nodes: {len(initial_node_id_list)} nodes")
                # Create explainer with current parameters
                explainer = create_candidate_explainer(
                    initial_node_id_list=initial_node_id_list,
                    candidate_set=candidate_set, model=downstream_model,
                    nodeid_index_dict=nodeid_index_dict,
                    ppi_edges=ppi_edges, 
                    num_entity=num_entity, device=device,
                    epochs=current_epochs, lr=current_lr)
                # Reset CUDA cache before graph generation
                torch.cuda.empty_cache()
                # ******************************** Generate the best explanation graph **********************************
                # Create sample-specific subdirectory
                sample_plots_dir = os.path.join(plots_dir, f"sample_{sample_id}")
                if not os.path.exists(sample_plots_dir):
                    os.makedirs(sample_plots_dir)
                # Define output paths based on sample_id and retry count
                origin_output_path = os.path.join(sample_plots_dir, f'origin_retry{retry_count}')
                origin_output_com_path = os.path.join(sample_plots_dir, f'com_origin_retry{retry_count}')
                conn_output_path = os.path.join(sample_plots_dir, f'conn_retry{retry_count}')
                conn_output_com_path = os.path.join(sample_plots_dir, f'com_retry{retry_count}')
                print(f"Attempt {retry_count+1}: Generating graph with {len(initial_node_id_list)} initial nodes, "
                    f"{current_samples} samples, {current_nodes} max nodes, {current_steps} steps")
                # Generate the best explanation graph with current parameters
                best_sampled_graph, completed_best_sampled_graph, best_connected_graph, best_conn_completed_graph = generate_best_graph(
                    explainer=explainer, target_class=1, device=device, nodeid_index_dict=nodeid_index_dict,
                    ppi_edges=ppi_edges, num_samples=current_samples, num_nodes=current_nodes, max_steps=current_steps, 
                    origin_output_path=origin_output_path, origin_output_com_path=origin_output_com_path,
                    conn_output_path=conn_output_path, conn_output_com_path=conn_output_com_path, mask_plot=True) # mask_plot=False will generatr all plots
                # Validate returned graphs
                if not hasattr(completed_best_sampled_graph, 'node_id') or len(completed_best_sampled_graph.node_id) == 0:
                    raise ValueError("Original generated graph has no nodes")
                if not hasattr(best_conn_completed_graph, 'node_id') or len(best_conn_completed_graph.node_id) == 0:
                    raise ValueError("Generated graph has no nodes")
                # Extract the node id list of the completed best sampled graph
                completed_best_sampled_graph_bmgc_id_list = completed_best_sampled_graph.node_id
                # Extract the node id list of the best connected graph
                best_conn_completed_graph_bmgc_id_list = best_conn_completed_graph.node_id
                # Check if we have at least some meaningful number of nodes in the graph
                if len(completed_best_sampled_graph_bmgc_id_list) < 3:
                    raise ValueError(f"Generated graph has only {len(completed_best_sampled_graph_bmgc_id_list)} nodes, which is insufficient")
                # Check if we have at least some meaningful number of nodes in the graph
                if len(best_conn_completed_graph_bmgc_id_list) < 3:
                    raise ValueError(f"Generated graph has only {len(best_conn_completed_graph_bmgc_id_list)} nodes, which is insufficient")
                # Convert bmgc_id_list to corresponding hgnc_symbol for LLM
                completed_best_sampled_graph_bmgc_id_hgnc_dict, completed_best_sampled_graph_hgnc_list = bmgc_pt_id_to_hgnc(completed_best_sampled_graph_bmgc_id_list, bmgc_protein_df)
                best_conn_completed_graph_bmgc_id_hgnc_dict, best_conn_completed_graph_hgnc_list = bmgc_pt_id_to_hgnc(best_conn_completed_graph_bmgc_id_list, bmgc_protein_df)
                
                # [completed_best_sampled_graph]
                # Build up the completed_best_sampled_graph (completed_best_sampled_graph.x.shape[0]) and node_id dict
                completed_best_sampled_graph_index_nodeid_dict = {}
                completed_best_sampled_graph_index_origin_index_dict = {}
                completed_best_sampled_graph_index_hgnc_dict = {}
                for i, node_id in enumerate(completed_best_sampled_graph_bmgc_id_list):
                    completed_best_sampled_graph_index_nodeid_dict[i] = node_id
                    completed_best_sampled_graph_index_origin_index_dict[i] = nodeid_index_dict[node_id]
                    if node_id in completed_best_sampled_graph_bmgc_id_hgnc_dict:
                        completed_best_sampled_graph_index_hgnc_dict[i] = completed_best_sampled_graph_bmgc_id_hgnc_dict[node_id]
                # [best_conn_completed_graph]
                # Build up the best_conn_completed_graph (best_sampled_graph.x.shape[0]) and node_id dict
                best_conn_completed_graph_index_nodeid_dict = {}
                best_conn_completed_graph_index_origin_index_dict = {}
                best_conn_completed_graph_index_hgnc_dict = {}
                for i, node_id in enumerate(best_conn_completed_graph_bmgc_id_list):
                    best_conn_completed_graph_index_nodeid_dict[i] = node_id
                    best_conn_completed_graph_index_origin_index_dict[i] = nodeid_index_dict[node_id]
                    if node_id in best_conn_completed_graph_bmgc_id_hgnc_dict:
                        best_conn_completed_graph_index_hgnc_dict[i] = best_conn_completed_graph_bmgc_id_hgnc_dict[node_id]

                # [completed_best_sampled_graph]
                # Convert best_conn_completed_graph edge_index to hgnc_symbol relation pairs
                hgnc_edges, edge_text_descriptions = convert_edges_to_hgnc_symbols(completed_best_sampled_graph.edge_index, completed_best_sampled_graph_index_hgnc_dict, completed_best_sampled_graph_index_nodeid_dict)
                hgnc_additional_edges, additional_edge_text_descriptions = convert_edges_to_hgnc_symbols(completed_best_sampled_graph.additional_edge_index, completed_best_sampled_graph_index_hgnc_dict, completed_best_sampled_graph_index_nodeid_dict)
                # # Remove edges that contains "Unknown"
                # hgnc_edges = [edge for edge in hgnc_edges if "Unknown" not in edge[0] and "Unknown" not in edge[1]]
                # hgnc_additional_edges = [edge for edge in hgnc_additional_edges if "Unknown" not in edge[0] and "Unknown" not in edge[1]]
                # Convert the edge_index into the original index
                original_edge_index = convert_to_original_index(completed_best_sampled_graph.edge_index, completed_best_sampled_graph_index_origin_index_dict)
                original_additional_edge_index = convert_to_original_index(completed_best_sampled_graph.additional_edge_index, completed_best_sampled_graph_index_origin_index_dict)
                # Define hgnc plot output path on sample_id and retry count
                hgnc_plot_path = os.path.join(sample_plots_dir, f'hgnc_plot_retry{retry_count}')
                hgnc_com_plot_path = os.path.join(sample_plots_dir, f'hgnc_com_plot_retry{retry_count}')
                visualize_hgnc_edges(hgnc_edges, hgnc_additional_edges, hgnc_plot_path, hgnc_com_plot_path)
                completed_best_sampled_graph_edge_index = completed_best_sampled_graph.edge_index
                completed_best_sampled_graph_additional_edge_index = completed_best_sampled_graph.additional_edge_index

                # [best_conn_completed_graph]
                # Convert best_conn_completed_graph edge_index to hgnc_symbol relation pairs
                conn_hgnc_edges, conn_edge_text_descriptions = convert_edges_to_hgnc_symbols(best_conn_completed_graph.edge_index, best_conn_completed_graph_index_hgnc_dict, best_conn_completed_graph_index_nodeid_dict)
                conn_hgnc_additional_edges, conn_additional_edge_text_descriptions = convert_edges_to_hgnc_symbols(best_conn_completed_graph.additional_edge_index, best_conn_completed_graph_index_hgnc_dict, best_conn_completed_graph_index_nodeid_dict)
                # # Remove edges that contains "Unknown"
                # conn_hgnc_edges = [edge for edge in conn_hgnc_edges if "Unknown" not in edge[0] and "Unknown" not in edge[1]]
                # conn_hgnc_additional_edges = [edge for edge in conn_hgnc_additional_edges if "Unknown" not in edge[0] and "Unknown" not in edge[1]]
                # Convert the edge_index into the original index
                conn_original_edge_index = convert_to_original_index(best_conn_completed_graph.edge_index, best_conn_completed_graph_index_origin_index_dict)
                conn_original_additional_edge_index = convert_to_original_index(best_conn_completed_graph.additional_edge_index, best_conn_completed_graph_index_origin_index_dict)
                # Define hgnc plot output path on sample_id and retry count
                hgnc_conn_plot_path = os.path.join(sample_plots_dir, f'hgnc_conn_plot_retry{retry_count}')
                hgnc_conn_com_plot_path = os.path.join(sample_plots_dir, f'hgnc_conn_com_plot_retry{retry_count}')
                visualize_hgnc_edges(conn_hgnc_edges, conn_hgnc_additional_edges, hgnc_conn_plot_path, hgnc_conn_com_plot_path)
                best_conn_completed_graph_edge_index = best_conn_completed_graph.edge_index
                best_conn_completed_graph_additional_edge_index = best_conn_completed_graph.additional_edge_index

                # Print graph description for debugging
                print(f"🔍 ** Identified Subsignnaling Gene Regulatory Network from SubGraph Generator **\n"
                    f"- Involved genes in the subgraph: {', '.join(completed_best_sampled_graph_hgnc_list)}\n" # best_conn_completed_graph_hgnc_list / completed_best_sampled_graph_hgnc_list
                    f"- The following signaling path was inferred to represent a likely cascade:\n"
                    f"{edge_text_descriptions}\n\n") # conn_edge_text_descriptions / edge_text_descriptions
                # Graph generation successful - set flag to exit loop
                graph_generation_successful = True
                print(f"✅ Graph generation successful on attempt {retry_count+1}/{max_retries}")

            except Exception as e:
                # Increment retry counter
                retry_count += 1
                # Handle exception
                print(f"⚠️ Error in Graph generation for sample {sample_id} (attempt {retry_count}/{max_retries}): {str(e)}")
                # Clear CUDA cache
                torch.cuda.empty_cache()
                # If we've reached max retries, use fallback
                if retry_count >= max_retries:
                    print(f"❌ All {max_retries} attempts failed for sample {sample_id}. Using fallback...")
                    # Log all attempts for debugging
                    print("Attempt summary:")
                    for idx, params in enumerate(attempt_params):
                        print(f"  Attempt {idx+1}: epochs={params['epochs']}, lr={params['lr']}, "
                            f"samples={params['samples']}, nodes={params['nodes']}, steps={params['steps']}")
                    
                    # [completed_best_sampled_graph]
                    # Use LLM first step results as fallback
                    completed_best_sampled_graph_bmgc_id_list = []
                    completed_best_sampled_graph_hgnc_list = []
                    completed_best_sampled_graph_edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
                    completed_best_sampled_graph_additional_edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
                    # [best_conn_completed_graph]
                    # Use LLM first step results as fallback
                    best_conn_completed_graph_bmgc_id_list = []
                    best_conn_completed_graph_hgnc_list = []
                    best_conn_completed_graph_edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
                    best_conn_completed_graph_additional_edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
                    # Edge descriptions
                    edge_text_descriptions = "Graph generation failed after multiple attempts."
                    conn_edge_text_descriptions = "Connected graph generation failed after multiple attempts."
                    # Create empty structures for downstream code
                    hgnc_edges = []
                    conn_hgnc_edges = []
                    hgnc_additional_edges = []
                    conn_hgnc_additional_edges = []
                    original_edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
                    conn_original_edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
                    original_additional_edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
                    conn_original_additional_edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
                    break
        
        # ***************************************************************************************************************************
        # ***************************************************** LLM 2nd STEP ********************************************************
        # ***************************************************************************************************************************
        # ************************************** LLM Reasoning with Prompt ************************************
        prompt2 = refine_with_graph_context_prompt(sample_data=sample_info, 
                                                   initial_reasoning=initial_reasoning, 
                                                   prev_proteins=llm_1st_step_response_hgnc_list,
                                                   graph_context=edge_text_descriptions, #  (can be replaced with additional_edge_text_descriptions)
                                                   graph_proteins=completed_best_sampled_graph_hgnc_list) # best_conn_completed_graph_hgnc_list
        refined_reasoning = run_llm(llm_pipeline, prompt2)
        # Truncate the refined reasoning after symbol "</s>"
        if "</s>" in refined_reasoning: 
            refined_reasoning = refined_reasoning.split("</s>")[0]
        elif "</p>" in refined_reasoning:
            refined_reasoning = refined_reasoning.split("</p>")[0]
        elif "</INST>" in refined_reasoning:
            refined_reasoning = refined_reasoning.split("</INST>")[0]
        elif "[/Output]" in refined_reasoning:
            refined_reasoning = refined_reasoning.split("[/Output]")[0]
        elif "[/Instruction]" in refined_reasoning:
            refined_reasoning = refined_reasoning.split("[/Instruction]")[0]
        elif "[/INST]" in refined_reasoning:
            refined_reasoning = refined_reasoning.split("[/INST]")[0]
        print("Step 2 - Refined LLM Reasoning:\n", refined_reasoning)
        # ********************************** Convert LLM Protein to Protein Node ********************************
        if ner_model_type == "GLiNER":
            # Use GLiNER model to extract entities
            refined_entities = gliner_model.predict_entities(refined_reasoning, labels)
            # Reduce the duplicated entities
            unique_refined_entities = dict({(entity["text"], entity["label"]) for entity in refined_entities})
            # Convert the unique entities to a list of proteins
            llm_2nd_step_response_hgnc_list = list(unique_refined_entities.keys())
            print("\nStep 2 - LLM Gene/Protein List:\n", llm_2nd_step_response_hgnc_list)
        elif ner_model_type == "ChatGPT":
            # Use ChatGPT API to extract entities
            llm_2nd_step_response_hgnc_list = extract_proteins_with_chatgpt(refined_reasoning, api_key, api_url)
            print("\nStep 2 - LLM Gene/Protein List:\n", llm_2nd_step_response_hgnc_list)

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
        kg_metrics = calculate_metrics(completed_best_sampled_graph_hgnc_list, crispr_answer) # completed_best_sampled_graph_hgnc_list
        llm_2nd_metrics = calculate_metrics(llm_2nd_step_response_hgnc_list, crispr_answer)
        
        # Build comprehensive results dictionary and print metrics
        results_dict = build_and_print_results(
            initial_reasoning=initial_reasoning,
            llm_1st_step_response_hgnc_list=llm_1st_step_response_hgnc_list,

            completed_best_sampled_graph_bmgc_id_list=completed_best_sampled_graph_bmgc_id_list,
            completed_best_sampled_graph_edge_index=completed_best_sampled_graph_edge_index,
            completed_best_sampled_graph_additional_edge_index=completed_best_sampled_graph_additional_edge_index,
            completed_best_sampled_graph_hgnc_list=completed_best_sampled_graph_hgnc_list, # used in 2nd step reasoning

            best_conn_completed_graph_bmgc_id_list=best_conn_completed_graph_bmgc_id_list,
            best_conn_completed_graph_edge_index=best_conn_completed_graph_edge_index,
            best_conn_completed_graph_additional_edge_index=best_conn_completed_graph_additional_edge_index,
            best_conn_completed_graph_hgnc_list=best_conn_completed_graph_hgnc_list, # alternative: used in 2nd step reasoning

            hgnc_edges=hgnc_edges,
            hgnc_additional_edges=hgnc_additional_edges,
            original_edge_index=original_edge_index,
            original_additional_edge_index=original_additional_edge_index,

            conn_hgnc_edges=conn_hgnc_edges,
            conn_hgnc_additional_edges=conn_hgnc_additional_edges,
            conn_original_edge_index=conn_original_edge_index,
            conn_original_additional_edge_index=conn_original_additional_edge_index,

            edge_text_descriptions=edge_text_descriptions,
            conn_edge_text_descriptions=conn_edge_text_descriptions,

            refined_reasoning=refined_reasoning,
            llm_2nd_step_response_hgnc_list=llm_2nd_step_response_hgnc_list,
            llm_1st_metrics=llm_1st_metrics,
            kg_metrics=kg_metrics,
            llm_2nd_metrics=llm_2nd_metrics,
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
    
    # Check device and select the one with most available memory
    if hasattr(args, 'device') and args.device < 0:
        device = 'cpu'
        print("Using CPU (forced by args)")
    else:
        device = select_best_gpu_device()

    import os
    # Add debugging flags at the top of your script
    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Force check for cuda:1 in MIG environment
    preferred_device = "cuda:1"
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    is_mig_environment = 'MIG-' in cuda_visible
    
    if is_mig_environment:
        print(f"🔧 MIG environment detected: {cuda_visible}")
        # Check how many MIG devices are available
        mig_devices = cuda_visible.split(',')
        print(f"Available MIG devices: {len(mig_devices)}")
        
        if torch.cuda.device_count() > 1:
            try:
                # Test if cuda:1 exists and is accessible
                torch.cuda.set_device(1)
                test_tensor = torch.tensor([1.0], device=preferred_device)
                props = torch.cuda.get_device_properties(1)
                allocated = torch.cuda.memory_allocated(1) / (1024**3)
                total = props.total_memory / (1024**3)
                free = total - allocated
                
                print(f"✅ MIG Device cuda:1 found and accessible:")
                print(f"  Name: {props.name}")
                print(f"  Total Memory: {total:.1f} GB")
                print(f"  Available Memory: {free:.1f} GB")
                
                del test_tensor
                torch.cuda.empty_cache()
                device = preferred_device
                print(f"Using forced MIG device: {device}")
                
            except Exception as e:
                print(f"❌ Cannot access cuda:1 in MIG environment: {e}")
                print(f"Available devices: {[f'cuda:{i}' for i in range(torch.cuda.device_count())]}")
                print(f"Falling back to detected device: {device}")
        else:
            print(f"⚠️  Only {torch.cuda.device_count()} MIG device(s) available")
            print(f"Cannot use cuda:1, using detected device: {device}")
    else:
        # Non-MIG environment - normal cuda:1 check
        if torch.cuda.device_count() > 1:
            try:
                torch.cuda.set_device(1)
                test_tensor = torch.tensor([1.0], device=preferred_device)
                del test_tensor
                device = preferred_device
                print(f"✅ Using preferred device: {device}")
            except Exception as e:
                print(f"❌ Cannot use {preferred_device}: {e}")
                print(f"Using detected device: {device}")
        else:
            print(f"⚠️  Only {torch.cuda.device_count()} GPU(s) available")
            print(f"Using detected device: {device}")
    
    print(f"Final device selection: {device}")

    # Train the model
    train(args, device)