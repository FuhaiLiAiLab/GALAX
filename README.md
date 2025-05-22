# GALAX
Graph Augmented LAnguage Model with eXplainbility

## 1. Download BioMedical Knowledge Graph
Check the website at [BioMedGraphica](https://github.com/FuhaiLiAiLab/BioMedGraphica) and download it to the path './BMG'

Then, run the code in each folder with entity and relation, with acquiring entities and relations.

## 2. Collect the DepMap data
Check the website at [DepMap](https://depmap.org/portal/data_page/?tab=overview) and download following links:

| **File Type**           | **File Name**                          | **Download Site**                                                                 |
|-------------------------|----------------------------------------|-----------------------------------------------------------------------------------|
| Promoter feature        | CCLE_RRBS_TSS1kb_20181022.txt          | [Link](https://depmap.org/portal/data_page/?tab=allData)                         |
| Gene feature            | OmicsCNGene.csv                        | [Link](https://depmap.org/portal/data_page/?tab=allData)                         |
| Transcript feature      | See footnote$^{\dag}$                  | [Link](https://depmap.org/portal/data_page/?tab=allData)                         |
| Protein feature         | protein_quant_current_normalized.csv   | [Link](https://depmap.org/portal/data_page/?tab=allData)                         |
| CRISPR gene effect      | CRISPRGeneEffect.csv                   | [Link](https://depmap.org/portal/data_page/?tab=allData)                         |
| Cell line annotation    | Table_S1_Sample_Information.xlsx       | [Link](https://depmap.org/portal/data_page/?tab=allData)                         |
| Cell line annotation    | cellosaurus.obo                        | [Link](https://ftp.expasy.org/databases/cellosaurus/cellosaurus.obo)            |
| Cell line status        | cell-lines-in-Non-Cancerous.csv        | [Link](https://depmap.org/portal/context/Non-Cancerous)                          |

And put them under the folder ./BMG/raw

## 3. Process the data
Just run the ./BMG/process.ipynb to get the integrated multi-omics data and Target-QA data with 'multi_sample_qa_info_k{k}_bm{top_bm}.json' (e.g., k=10, top_bm=100). And run the ./BMG/medtune.ipynb to get the 'mixed_description.jsonl'.

## 4. Pretrain the language model
Pretrain the llama3-8B-Instruct with 2 NVIDIA H100 (80G) GPUs with
```
accelerate launch --num_processes=4 pretrain_llama_fa.py
```

## 5. Pretrain the graph foundation model
### 5.1 Capturing the edge mechanism
Run the pretraining model
```
python motasg_pretrain.py
```

### 5.2 Pretrain the disease status classification
Run the classification pretraining model
```
python motasg_train.py
```

## 6. Training the GALAX
### 6.1 Split the data 
```
python qa_train_test.py
```


### 6.2 Run the model
Pretrain the initial answering
```
python finetune_llama.py
```

Then, run the GALAX reasoning with explainable subgraph
```
python GALAX.py
```

Then finetune the 2nd stage / final answering
```
python finetune_llama_2nd_step.py
```


### 6.3 Other baseline models
The LLM based models
```
python plainllm_omics.py
python plainllm_omicskg.py
python bmgcllm_omics.py
python bmgcllm_omicskg.py
python qallm_omics.py
python qallm_omicskg.py
```

The Graph-based model
```
python motasg_crispr_ko.py
python motasg_crispr_ko_analysis.py
```

The Graph Language model
```
python g-Retriever.py
```