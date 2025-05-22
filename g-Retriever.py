import argparse
import gc
import math
import os
import os.path as osp
import re
import time
import json
import datetime
import numpy as np
from pathlib import Path
import requests

import pandas as pd
import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from torch_geometric import seed_everything
from torch_geometric.loader import DataLoader

import torch.nn as nn
from torch_geometric.nn import GCNConv

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

from torch_geometric.typing import (
    Adj,
    Size,
    NoneType,
    OptTensor,
    OptPairTensor,
    PairTensor,
    SparseTensor,
    torch_sparse,
)

import warnings
from contextlib import nullcontext
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

try:
    from transformers.tokenization_utils_base import BatchEncoding
except ImportError:
    BatchEncoding = Dict

BOS = '<s>[INST]'
EOS_USER = '[/INST]'
EOS = '[/s]'
IGNORE_INDEX = -100
MAX_TXT_LEN = 4000
MAX_NEW_TOKENS = 4000
PAD_TOKEN_ID = 0
PADDING_SIDE = 'left'


def get_llm_kwargs(required_memory: int, dtype=torch.dtype) -> Dict[str, Any]:
    torch.cuda.empty_cache()

    gpu_memory: List[int] = []
    for i in range(torch.cuda.device_count()):
        gpu_memory.append(torch.cuda.mem_get_info(i)[0] // 1024**3)
        # Use the minimum number of GPUs to fit the LLM on.
        if sum(gpu_memory) >= required_memory:
            break

    if sum(gpu_memory) < required_memory:
        gpu_memory = []  # If not enough VRAM, use pure CPU.

    kwargs = dict(revision='main')
    if len(gpu_memory) > 0:
        kwargs['max_memory'] = {
            i: f'{memory}GiB'
            for i, memory in enumerate(gpu_memory)
        }
        kwargs['low_cpu_mem_usage'] = True
        kwargs['device_map'] = 'auto'
        kwargs['torch_dtype'] = dtype

    return kwargs

def arg_parse():
    parser = argparse.ArgumentParser()

    # pre-training loading parameters
    parser.add_argument('--layer', nargs='?', default='gat', help='GNN layer, (default: gat)')
    parser.add_argument('--encoder_activation', nargs='?', default='leaky_relu', help='Activation function for GNN encoder, (default: leaky_relu)')

    parser.add_argument('--num_omic_feature', type=int, default=1, help='Omic feature size. (default: 1)')
    
    parser.add_argument('--input_dim', type=int, default=1, help='Input feature dimension. (default: 1)')
    parser.add_argument('--encoder_channels', type=int, default=8, help='Channels of GNN encoder layers. (default: 8)')
    parser.add_argument('--hidden_channels', type=int, default=8, help='Channels of hidden representation. (default: 8)')
    parser.add_argument('--decoder_channels', type=int, default=4, help='Channels of decoder layers. (default: 4)')

    parser.add_argument('--encoder_layers', type=int, default=2, help='Number of layers for encoder. (default: 2)')
    parser.add_argument('--internal_encoder_layers', type=int, default=4, help='Number of layers for internal encoder. (default: 4)')
    parser.add_argument('--decoder_layers', type=int, default=2, help='Number of layers for decoders. (default: 2)')
    parser.add_argument('--encoder_dropout', type=float, default=0.2, help='Dropout probability of encoder. (default: 0.2)')
    parser.add_argument('--decoder_dropout', type=float, default=0.2, help='Dropout probability of decoder. (default: 0.2)')

    parser.add_argument('--p', type=float, default=0.0001, help='Mask ratio or sample ratio for MaskEdge')

    parser.add_argument('--bn', action='store_true', help='Whether to use batch normalization for GNN encoder. (default: False)')
    parser.add_argument('--l2_normalize', action='store_true', help='Whether to use l2 normalize output embedding. (default: False)')
    parser.add_argument('--graphclas_weight_decay', type=float, default=1e-3, help='weight_decay for node classification training. (default: 1e-3)')

    parser.add_argument('--save_path', nargs='?', default='./Checkpoints/pretrained_models_gat/pretrained_motasg_foundation.pt', help='save path for model. (default: pretrained_motasg_foundation.pt)')
    
    # downstream task parameters
    parser.add_argument('--train_sample_ratio', type=float, default=0.2, help='Sampling ratio of training data. (default: 0.2)')
    parser.add_argument('--training_sample_random_seed', type=int, default=2025, help='Random seed for sampling training data. (default: 2025)')

    parser.add_argument('--text_lm_model_path', nargs='?', default='dmis-lab/biobert-v1.1', help='Path to the pretrained language model. (default: dmis-lab/biobert-v1.1)')
    parser.add_argument('--train_text', default=False, help='Whether to train the text encoder. (default: False)')
    parser.add_argument('--task', nargs='?', default='class', help='Task for training downstream tasks. (default: class)')
    parser.add_argument('--name', nargs='?', default='DepMap', help='Name for dataset.')
    parser.add_argument('--num_class', type=int, default=2, help='Number of classes for classification. (default: 2)')

    parser.add_argument('--train_weight_decay', type=float, default=1e-15, help='Weight decay for Adam optimizer. (default: 1e-15)')
    parser.add_argument('--train_encoder_dropout', type=float, default=0.1, help='Dropout probability of encoder. (default: 0.1)')

    parser.add_argument('--train_layer', nargs='?', default='gat', help='GNN layer, (default: gat)')
    parser.add_argument('--train_internal_encoder_layers', type=int, default=3, help='Number of layers for internal encoder. (default: 3)')
    parser.add_argument('--train_encoder_layers', type=int, default=2, help='Number of layers for encoder. (default: 2)')

    parser.add_argument('--pre_input_dim', type=int, default=8, help='Input feature dimension for pretraining. (default: 8)')
    parser.add_argument('--train_fusion_dim', type=int, default=1, help='Fusion feature dimension for training. (default: 1)')
    parser.add_argument('--train_hidden_dim', type=int, default=8, help='Hidden feature dimension for training. (default: 8)')
    parser.add_argument('--train_output_dim', type=int, default=8, help='Output feature dimension for training. (default: 8)')

    parser.add_argument('--train_linear_input_dim', type=int, default=8, help='Input feature dimension for training. (default: 8)')
    parser.add_argument('--train_linear_hidden_dim', type=int, default=32, help='Hidden feature dimension for training. (default: 32)')
    parser.add_argument('--train_linear_output_dim', type=int, default=16, help='Output feature dimension for training. (default: 16)')

    parser.add_argument('--train_save_path', nargs='?', default='./MOTASG_Results/DepMap/MOTASG_Class_gat_gat/epoch_50_best/best_train_model.pt', help='Path to save the trained model.')

    # G-Retriever model parameters
    parser.add_argument('--device', type=int, default=0, help='Device ID to use for training (default: 0)')
    parser.add_argument('--lm_emb_dim', type=int, default=1, help='Text embedding dimension (default: 1)')
    parser.add_argument('--omics_feature_dim', type=int, default=1, help='Omics feature dimension (default: 1)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training (default: 1)')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of epochs (default: 1)')
    parser.add_argument('--learning_rate', type=float, default=5e-6, help='Learning rate (default: 5e-6)')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (default: 1e-5)')
    parser.add_argument('--train_json', type=str, default='./QA_Data/multi_sample_qa_info_k100_bm100_tr.json', 
                       help='Training data JSON path')
    parser.add_argument('--test_json', type=str, default='./QA_Data/multi_sample_qa_info_k100_bm100_te.json', 
                       help='Testing data JSON path')
    parser.add_argument('--save_dir', type=str, default='./Checkpoints/gretriever', 
                       help='Directory to save models')
    parser.add_argument('--output_result_dir', type=str, default='./QA_Results', 
                      help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')

    parser.add_argument('--model_name', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct', help='LLM model name (default: meta-llama/Meta-Llama-3-8B-Instruct)')
    parser.add_argument('--base_model_name', type=str, default=None, help='Base LLM model name for tokenizer (default: None)')
    parser.add_argument('--use_lora', action='store_false', help='Whether to use LoRA for fine-tuning (default: False)')
    parser.add_argument('--mlp_out_channels', type=int, default=4096, help='Size of graph embedding after projection (default: 4096)')
    parser.add_argument('--mlp_out_tokens', type=int, default=1, help='Number of LLM prefix tokens for GNN output (default: 1)')
    parser.add_argument('--gnn_hidden_channels', type=int, default=8, help='Hidden channels for SimpleGNN (default: 8)')
    parser.add_argument('--gnn_out_channels', type=int, default=8, help='Output channels for SimpleGNN (default: 8)')
    parser.add_argument('--max_out_tokens', type=int, default=4000, help='Maximum tokens for LLM to generate (default: 4000)')

    return parser.parse_args()


from typing import List, Optional

import torch
from torch import Tensor

from torch_geometric.nn.nlp.llm import BOS, LLM, MAX_NEW_TOKENS
from torch_geometric.utils import scatter


class LLM(torch.nn.Module):
    r"""A wrapper around a Large Language Model (LLM) from HuggingFace.

    Args:
        model_name (str): The HuggingFace model name, *e.g.*, :obj:`"llama2"`
            or :obj:`"gemma"`.
        num_params (int, optional): An integer representing how many parameters
            the HuggingFace model has, in billions. This is used to
            automatically allocate the correct number of GPUs needed, given the
            available GPU memory of your GPUs. If not specified, the number of
            parameters is determined using the `huggingface_hub` module.
        dtype (torch.dtype, optional): The data type to use for the LLM.
            (default :obj: `torch.bfloat16`)
    """
    def __init__(
        self,
        model_name: str,
        base_model_name: Optional[str] = None,
        num_params: Optional[int] = None,
        dtype: Optional[torch.dtype] = torch.bfloat16,
    ) -> None:
        super().__init__()

        self.model_name = model_name

        from transformers import AutoModelForCausalLM, AutoTokenizer

        if num_params is None:
            from huggingface_hub import get_safetensors_metadata
            safetensors_metadata = get_safetensors_metadata(model_name)
            param_count = safetensors_metadata.parameter_count
            num_params = list(param_count.values())[0] // 10**9

        # A rough heuristic on GPU memory requirements, e.g., we found that
        # LLAMA2 (7B parameters) fits on a 85GB GPU.
        required_memory = 85 * num_params / 7
        kwargs = get_llm_kwargs(required_memory, dtype)

        # ⚡ Fix: Always load with correct device map
        kwargs["device_map"] = "auto"
        kwargs["torch_dtype"] = dtype

        print(f"Setting up '{model_name}' with configuration: {kwargs}")

        # Set up tokenizer with explicit pad token configuration
        from os import path as osp

        if osp.exists(model_name):
            base_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

        self.tokenizer.padding_side = PADDING_SIDE
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        else:
            self.tokenizer.pad_token_id = PAD_TOKEN_ID

        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            **kwargs,
        )

        self.word_embedding = self.llm.model.get_input_embeddings()

        if 'max_memory' not in kwargs:  # Pure CPU:
            warnings.warn("LLM is being used on CPU, which may be slow")
            self.device = torch.device('cpu')
            self.autocast_context = nullcontext()
        else:
            self.device = self.llm.device
            self.autocast_context = torch.amp.autocast('cuda', dtype=dtype)

    def _encode_inputs(
        self,
        question: List[str],
        context: Optional[List[str]] = None,
    ) -> tuple:
        batch_size = len(question)
        questions = self.tokenizer(question, add_special_tokens=False)
        if context is not None:
            context = self.tokenizer(context, add_special_tokens=False)

        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        bos_token = self.tokenizer(
            BOS,
            add_special_tokens=False,
            return_tensors='pt',
        ).input_ids[0].to(self.device)
        bos_embeds = self.word_embedding(bos_token)
        pad_token = torch.tensor(self.tokenizer.pad_token_id,
                                 device=self.device)
        pad_embeds = self.word_embedding(pad_token).unsqueeze(0)
        return (batch_size, questions, context, eos_user_tokens, bos_embeds,
                pad_embeds)

    def _label_input_ids(
        self,
        i: int,
        label: BatchEncoding,
        eos_tokens: BatchEncoding,
    ) -> List[int]:
        label_input_ids = label.input_ids[i][:MAX_NEW_TOKENS]
        label_input_ids = label_input_ids + eos_tokens.input_ids
        return label_input_ids

    def _input_ids(
        self,
        i: int,
        context: BatchEncoding,
        question: BatchEncoding,
        eos_user_tokens: BatchEncoding,
    ) -> List[int]:
        input_ids: List[int] = []
        if context is not None:
            input_ids += context.input_ids[i][:MAX_TXT_LEN]
        input_ids += question.input_ids[i]
        input_ids += eos_user_tokens.input_ids
        return input_ids

    def _inputs_embeds(
        self,
        i: int,
        input_ids: List[int],
        bos_embeds: Tensor,
        embedding: Optional[List[Tensor]] = None,
    ) -> Tensor:
        inputs_embeds = self.word_embedding(
            torch.tensor(input_ids, device=self.device))

        to_cat = [bos_embeds]
        if embedding is not None and embedding[i] is not None:
            to_cat.append(embedding[i])
        to_cat.append(inputs_embeds)
        return torch.cat(to_cat, dim=0).to(self.device)

    def _append_embeds(
        self,
        inputs_embeds: Tensor,
        batch_inputs_embeds: List[Tensor],
        batch_attention_mask: List[List[int]],
        label_input_ids: List[int] = None,
        batch_label_input_ids: Optional[List[List[int]]] = None,
    ) -> tuple:
        batch_inputs_embeds.append(inputs_embeds)
        batch_attention_mask.append([1] * inputs_embeds.size(0))
        if label_input_ids is not None:
            pad = inputs_embeds.size(0) - len(label_input_ids)
            label_input_ids = [IGNORE_INDEX] * pad + label_input_ids
            batch_label_input_ids.append(label_input_ids)
        return batch_inputs_embeds, batch_attention_mask, batch_label_input_ids

    def _pad_embeds(
        self,
        pad_embeds: Tensor,
        batch_inputs_embeds: List[Tensor],
        batch_attention_mask: List[List[int]],
        batch_label_input_ids: Optional[List[List[int]]] = None,
    ) -> tuple:
        max_length = max([x.size(0) for x in batch_inputs_embeds])
        batch_size = len(batch_inputs_embeds)
        for i in range(batch_size):
            pad = max_length - batch_inputs_embeds[i].size(0)
            batch_inputs_embeds[i] = torch.cat([
                pad_embeds.repeat(pad, 1),
                batch_inputs_embeds[i],
            ])
            batch_attention_mask[i] = [0] * pad + batch_attention_mask[i]
            if batch_label_input_ids is not None:
                tmp = [IGNORE_INDEX] * pad + batch_label_input_ids[i]
                batch_label_input_ids[i] = tmp
        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0)
        attention_mask = torch.tensor(batch_attention_mask, device=self.device)
        label_input_ids = None
        if batch_label_input_ids is not None:
            label_input_ids = torch.tensor(batch_label_input_ids,
                                           device=self.device)
        return inputs_embeds, attention_mask, label_input_ids

    def _get_embeds(
        self,
        question: List[str],
        context: Optional[List[str]] = None,
        embedding: Optional[List[Tensor]] = None,
        answer: Optional[List[str]] = None,
    ) -> tuple:
        (batch_size, question, context, eos_user_tokens, bos_embeds,
         pad_embeds) = self._encode_inputs(question, context)

        batch_label_input_ids = None
        if answer is not None:
            label = self.tokenizer(answer, add_special_tokens=False)
            eos_tokens = self.tokenizer(EOS, add_special_tokens=False)
            batch_label_input_ids = []

        batch_inputs_embeds = []
        batch_attention_mask = []
        for i in range(batch_size):
            input_ids = self._input_ids(i, context, question, eos_user_tokens)
            if answer is not None:
                label_input_ids = self._label_input_ids(i, label, eos_tokens)
                input_ids += label_input_ids
            else:
                label_input_ids = None

            inputs_embeds = self._inputs_embeds(i, input_ids, bos_embeds,
                                                embedding)

            (
                batch_inputs_embeds,
                batch_attention_mask,
                batch_label_input_ids,
            ) = self._append_embeds(
                inputs_embeds,
                batch_inputs_embeds,
                batch_attention_mask,
                label_input_ids,
                batch_label_input_ids,
            )

        inputs_embeds, attention_mask, label_input_ids = self._pad_embeds(
            pad_embeds, batch_inputs_embeds, batch_attention_mask,
            batch_label_input_ids)

        return inputs_embeds, attention_mask, label_input_ids

    def forward(
        self,
        question: List[str],
        answer: List[str],
        context: Optional[List[str]] = None,
        embedding: Optional[List[Tensor]] = None,
    ) -> Tensor:
        r"""The forward pass.

        Args:
            question (list[str]): The questions/prompts.
            answer (list[str]): The answers/labels.
            context (list[str], optional): Additional context to give to the
                LLM, such as textified knowledge graphs. (default: :obj:`None`)
            embedding (list[torch.Tensor], optional): RAG embedding
                tensors, *i.e.* the embedded form of :obj:`context`. Either
                :obj:`context` or :obj:`embedding` should be used, not
                both. (default: :obj:`None`)
        """
        inputs_embeds, attention_mask, label_input_ids = self._get_embeds(
            question, context, embedding, answer)

        with self.autocast_context:
            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=label_input_ids,
            )
        return outputs.loss

    @torch.no_grad()
    def inference(
        self,
        question: List[str],
        context: Optional[List[str]] = None,
        embedding: Optional[List[Tensor]] = None,
        max_tokens: Optional[int] = MAX_NEW_TOKENS,
    ) -> List[str]:
        r"""The inference pass.

        Args:
            question (list[str]): The questions/prompts.
            answer (list[str]): The answers/labels.
            context (list[str], optional): Additional context to give to the
                LLM, such as textified knowledge graphs. (default: :obj:`None`)
            embedding (list[torch.Tensor], optional): RAG embedding
                tensors, *i.e.* the embedded form of :obj:`context`. Either
                :obj:`context` or :obj:`embedding` should be used, not
                both. (default: :obj:`None`)
            max_tokens (int, optional): How many tokens for the LLM to
                generate. (default: :obj:`32`)
        """
        inputs_embeds, attention_mask, _ = self._get_embeds(
            question, context, embedding)

        bos_token = self.tokenizer(
            BOS,
            add_special_tokens=False,
        ).input_ids[0]

        with self.autocast_context:
            outputs = self.llm.generate(
                inputs_embeds=inputs_embeds,
                bos_token_id=bos_token,
                max_new_tokens=max_tokens,
                attention_mask=attention_mask,
                use_cache=True,
            )

        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.model_name})'


class GRetriever(torch.nn.Module):
    r"""The G-Retriever model from the `"G-Retriever: Retrieval-Augmented
    Generation for Textual Graph Understanding and Question Answering"
    <https://arxiv.org/abs/2402.07630>`_ paper.

    Args:
        llm (LLM): The LLM to use.
        gnn (torch.nn.Module): The GNN to use.
        use_lora (bool, optional): If set to :obj:`True`, will use LORA from
            :obj:`peft` for training the LLM, see
            `here <https://huggingface.co/docs/peft/en/index>`_ for details.
            (default: :obj:`False`)
        mlp_out_channels (int, optional): The size of each graph embedding
            after projection. (default: :obj:`4096`)
        mlp_out_tokens (int, optional): Number of LLM prefix tokens to
            reserve for GNN output. (default: :obj:`1`)

    .. warning::
        This module has been tested with the following HuggingFace models

        * :obj:`llm_to_use="meta-llama/Llama-2-7b-chat-hf"`
        * :obj:`llm_to_use="google/gemma-7b"`

        and may not work with other models. See other models at `HuggingFace
        Models <https://huggingface.co/models>`_ and let us know if you
        encounter any issues.

    .. note::
        For an example of using :class:`GRetriever`, see
        `examples/llm/g_retriever.py <https://github.com/pyg-team/
        pytorch_geometric/blob/master/examples/llm/g_retriever.py>`_.
    """
    def __init__(
        self,
        llm: LLM,
        gnn: torch.nn.Module,
        use_lora: bool = False,
        mlp_out_channels: int = 4096,
        mlp_out_tokens: int = 1,
    ) -> None:
        super().__init__()

        self.llm = llm
        self.gnn = gnn.to(self.llm.device)

        self.word_embedding = self.llm.word_embedding
        self.llm_generator = self.llm.llm
        if use_lora:
            from peft import (
                LoraConfig,
                get_peft_model,
                prepare_model_for_kbit_training,
            )
            self.llm_generator = prepare_model_for_kbit_training(
                self.llm_generator)
            lora_r: int = 8
            lora_alpha: int = 16
            lora_dropout: float = 0.05
            lora_target_modules = ['q_proj', 'v_proj']
            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias='none',
                task_type='CAUSAL_LM',
            )
            self.llm_generator = get_peft_model(self.llm_generator, config)

        mlp_hidden_channels = self.gnn.out_channels
        self.projector = torch.nn.Sequential(
            torch.nn.Linear(mlp_hidden_channels, mlp_hidden_channels),
            torch.nn.Sigmoid(),
            torch.nn.Linear(mlp_hidden_channels,
                            mlp_out_channels * mlp_out_tokens),
            torch.nn.Unflatten(-1, (mlp_out_tokens, mlp_out_channels)),
        ).to(self.llm.device)

    def encode(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        edge_attr: Optional[Tensor],
    ) -> Tensor:
        x = x.to(self.llm.device)
        edge_index = edge_index.to(self.llm.device)
        if edge_attr is not None:
            edge_attr = edge_attr.to(self.llm.device)
        batch = batch.to(self.llm.device)

        out = self.gnn(x, edge_index, edge_attr=edge_attr)
        return scatter(out, batch, dim=0, reduce='mean')

    def forward(
        self,
        question: List[str],
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        label: List[str],
        edge_attr: Optional[Tensor] = None,
        additional_text_context: Optional[List[str]] = None,
    ):
        r"""The forward pass.

        Args:
            question (List[str]): The questions/prompts.
            x (torch.Tensor): The input node features.
            edge_index (torch.Tensor): The edge indices.
            batch (torch.Tensor): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each element to a specific example.
            label (List[str]): The answers/labels.
            edge_attr (torch.Tensor, optional): The edge features (if supported
                by the GNN). (default: :obj:`None`)
            additional_text_context (List[str], optional): Additional context
                to give to the LLM, such as textified knowledge graphs.
                (default: :obj:`None`)
        """
        x = self.encode(x, edge_index, batch, edge_attr)
        x = self.projector(x)
        xs = x.split(1, dim=0)

        # Handle case where theres more than one embedding for each sample
        xs = [x.squeeze(0) for x in xs]

        # Handle questions without node features:
        batch_unique = batch.unique()
        batch_size = len(question)
        if len(batch_unique) < batch_size:
            xs = [
                xs[i] if i in batch_unique else None for i in range(batch_size)
            ]

        (
            inputs_embeds,
            attention_mask,
            label_input_ids,
        ) = self.llm._get_embeds(question, additional_text_context, xs, label)

        inputs_embeds = inputs_embeds.to(self.llm.device)
        attention_mask = attention_mask.to(self.llm.device)
        if label_input_ids is not None:
            label_input_ids = label_input_ids.to(self.llm.device)
        
        with self.llm.autocast_context:
            outputs = self.llm_generator(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=label_input_ids,
            )

        return outputs.loss


    @torch.no_grad()
    def inference(
        self,
        question: List[str],
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        edge_attr: Optional[Tensor] = None,
        additional_text_context: Optional[List[str]] = None,
        max_out_tokens: Optional[int] = MAX_NEW_TOKENS,
    ):
        r"""The inference pass.

        Args:
            question (List[str]): The questions/prompts.
            x (torch.Tensor): The input node features.
            edge_index (torch.Tensor): The edge indices.
            batch (torch.Tensor): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each element to a specific example.
            edge_attr (torch.Tensor, optional): The edge features (if supported
                by the GNN). (default: :obj:`None`)
            additional_text_context (List[str], optional): Additional context
                to give to the LLM, such as textified knowledge graphs.
                (default: :obj:`None`)
            max_out_tokens (int, optional): How many tokens for the LLM to
                generate. (default: :obj:`32`)
        """
        x = self.encode(x, edge_index, batch, edge_attr)
        x = self.projector(x)
        xs = x.split(1, dim=0)

        # Handle case where theres more than one embedding for each sample
        xs = [x.squeeze(0) for x in xs]

        # Handle questions without node features:
        batch_unique = batch.unique()
        batch_size = len(question)
        if len(batch_unique) < batch_size:
            xs = [
                xs[i] if i in batch_unique else None for i in range(batch_size)
            ]

        inputs_embeds, attention_mask, _ = self.llm._get_embeds(
            question, additional_text_context, xs)

        bos_token = self.llm.tokenizer(
            BOS,
            add_special_tokens=False,
        ).input_ids[0]

        with self.llm.autocast_context:
            outputs = self.llm_generator.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=max_out_tokens,
                attention_mask=attention_mask,
                bos_token_id=bos_token,
                use_cache=True,  # Important to set!
            )

        return self.llm.tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True,
        )


    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(\n'
                f'  llm={self.llm},\n'
                f'  gnn={self.gnn},\n'
                f')')


class SimpleGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.out_channels = out_channels

    def forward(self, x, edge_index, edge_attr=None):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        return x

def extract_proteins_with_chatgpt(text, api_key, api_url="https://api.openai.com/v1/chat/completions"):
    """Extract protein/gene mentions from text using ChatGPT API."""
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
    """Calculate evaluation metrics comparing predicted proteins with ground truth."""
    # Print lengths
    predicted_len = len(predicted_list)
    ground_truth_len = len(ground_truth_list)
    # Convert lists to sets for intersection/union operations
    predicted_set = set(predicted_list)
    ground_truth_set = set(ground_truth_list)
    # Calculate basic metrics
    overlap_set = ground_truth_set.intersection(predicted_set)
    overlap_count = len(overlap_set)
    precision = overlap_count / len(predicted_set) if len(predicted_set) > 0 else 0
    recall = overlap_count / len(ground_truth_set) if overlap_count > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    union_count = len(ground_truth_set.union(predicted_set))
    jaccard = overlap_count / union_count if union_count > 0 else 0
    # Precision at K
    precision_at_5 = precision_at_k(predicted_list, ground_truth_set, k=5)
    precision_at_10 = precision_at_k(predicted_list, ground_truth_set, k=10)
    return overlap_count, precision, recall, f1_score, jaccard, precision_at_5, precision_at_10

def prepare_gretriever_input(sample_info):
    """
    Prepare the three components for GRetriever input: question, additional_text_context, and label
    
    Args:
        sample_info (dict): Sample information from the QA dataset
        
    Returns:
        tuple: (question, additional_text_context, label)
    """
    cell_line_name = sample_info["cell_line_name"]
    disease = sample_info["disease"]

    # Component 1: Question (the instruction part)
    question = (
        f"Identify the 100 priority genes whose knockout causes the strongest negative effect "
        f"on the viability or proliferation of the {cell_line_name} cell line in the context of {disease}, "
        f"based on the highest relevance values derived from multi-omics datasets and knowledge graph information."
    )
    
    # Extract data for additional context
    top_gene = sample_info["input"]["top_k_gene"]["hgnc_symbols"]
    top_transcript = sample_info["input"]["top_k_transcript"]["hgnc_symbols"]
    top_protein = sample_info["input"]["top_k_protein"]["hgnc_symbols"]
    disease_protein = sample_info["input"]["knowledge_graph"]["disease_protein"]["hgnc_symbols"]
    protein_kg_relation = sample_info["input"]["knowledge_graph"]["protein_relationships"]
    
    # Truncate lists for input compactness with k items
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

    # Component 2: Additional text context (the input part)
    additional_text_context = (
        f"- Top 10 ranked genes with copy number values due to strong amplification: {', '.join(top_gene)}\n"
        f"- Top 10 ranked transcripts from transcriptomic profiling with high expression: {', '.join(top_transcript)}\n"
        f"- Top 10 ranked proteins from RPPA proteomics with high expression or activation: {', '.join(top_protein)}"
        f"- Disease-relevant proteins extracted from the biomedical knowledge graph: {', '.join(disease_protein)}\n"
        f"- Known protein-protein and disease-protein interactions from the knowledge graph: {', '.join(protein_kg_relation)}\n"
    )
    
    # Component 3: Label (the ground truth answer)
    ground_truth = sample_info["ground_truth_answer"]["top_bm_gene"]["hgnc_symbols"]
    
    # Format the label response
    label_response = (
        f"Based on the integrated multi-omics data and knowledge graph, I identified the 100 genes whose knockout "
        f"is predicted to have the most severe negative impact on the viability or proliferation of the "
        f"{sample_info['cell_line_name']} cell line in {sample_info['disease']}.\n\n"
        f"The prioritized gene list is as follows:\n\n"
    )
    
    # Add numbered gene list
    return question, additional_text_context, label_response

def save_results_to_json(sample_id, results_dict, output_dir, timestamp=None):
    """Save results to a JSON file."""
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for the filename if not provided
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a single filename for the entire run
    filename = f"gretriever_results_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Check if the file already exists
    if os.path.exists(filepath):
        # Load existing data
        with open(filepath, "r") as f:
            try:
                all_results = json.load(f)
            except json.JSONDecodeError:
                # File exists but is empty or corrupted
                print(f"Warning: Existing file {filepath} appears corrupted. Creating new file.")
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

def build_and_print_results(initial_reasoning, llm_response_hgnc_list, 
                           llm_metrics, crispr_answer, crispr_answer_dict):
    """Build results dictionary and print metrics."""
    # Unpack metrics
    llm_overlap_count, llm_precision, llm_recall, llm_f1_score, llm_jaccard, llm_precision_at_5, llm_precision_at_10 = llm_metrics
    
    # Build results dictionary
    results_dict = {
        "inputs": {
            "initial_reasoning": initial_reasoning
        },
        "outputs": {
            "llm_response_hgnc_list": llm_response_hgnc_list
        },
        "evaluation_results": {
            "llm_1st": {
                "precision": llm_precision,
                "recall": llm_recall,
                "f1_score": llm_f1_score,
                "overlap_count": llm_overlap_count,
                "jaccard": llm_jaccard,
                "precision@5": llm_precision_at_5,
                "precision@10": llm_precision_at_10
            }
        },
        "crispr_answer": crispr_answer_dict
    }

    # Print evaluation results
    print("\n********** Evaluation Results **********")
    print("\nLLM Results:")
    print(f"Precision: {llm_precision:.4f}")
    print(f"Recall: {llm_recall:.4f}")
    print(f"F1 Score: {llm_f1_score:.4f}")
    print(f"Overlap Count: {llm_overlap_count}/{len(crispr_answer)}")
    print(f"Jaccard Similarity: {llm_jaccard:.4f}")
    print(f"Precision at 5: {llm_precision_at_5:.4f}")
    print(f"Precision at 10: {llm_precision_at_10:.4f}")
    return results_dict

def train(args, gretriever, train_data, xAll_omics, pretrain_model, downstream_model, 
          name_embeddings, desc_embeddings, all_edge_index, internal_edge_index, 
          ppi_edge_index, device):
    """Training function for GRetriever model"""
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50)
    
    # Define optimizer
    optimizer = torch.optim.AdamW(
        gretriever.parameters(), 
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Create a unified timestamp for this run
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Training loop
    for epoch in range(args.num_epochs):
        gretriever.train()
        epoch_loss = 0
        
        # Get all training samples
        train_samples = list(train_data.items())
        
        # Process each sample individually
        for sample_idx, (sample_id, sample_info) in enumerate(tqdm(train_samples, desc=f"Epoch {epoch+1}/{args.num_epochs}")):
            # Generate components for GRetriever input
            question, additional_text_context, label = prepare_gretriever_input(sample_info)
            
            # Extract sample info
            selected_sample_index = sample_info["sample_dti_index"]
            selected_sample_x = xAll_omics[selected_sample_index, :].reshape(-1, args.omics_feature_dim)  # selected sample feature
            selected_sample_x_emb = pre_embed(pretrain_model, downstream_model, selected_sample_x, 
                                        name_embeddings, desc_embeddings, 
                                        all_edge_index, internal_edge_index, ppi_edge_index, device)
            
            # Ensure tensor has the correct data type
            selected_sample_x_emb = selected_sample_x_emb.to(device=gretriever.llm.device, dtype=torch.float32)
            sample_edge_index = all_edge_index.to(device=gretriever.llm.device)
            batch = torch.zeros(selected_sample_x.size(1), dtype=torch.long, device=gretriever.llm.device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            loss = gretriever(
                question=[question],
                x=selected_sample_x_emb,
                edge_index=sample_edge_index,
                batch=batch,
                label=[label],
                additional_text_context=[additional_text_context]
            )
            
            # Backward pass
            loss.backward()
            clip_grad_norm_(gretriever.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Print progress every few samples
            if (sample_idx + 1) % 10 == 0:
                print(f"  Sample {sample_idx + 1}/{len(train_samples)}, Current loss: {loss.item():.16f}")
        
        avg_loss = epoch_loss / len(train_samples)
        print(f"Epoch {epoch+1}/{args.num_epochs}, Average Loss: {avg_loss:.16f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(args.save_dir, f"gretriever_checkpoint_epoch{epoch+1}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': gretriever.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
    
    return gretriever, run_timestamp

def eval(args, gretriever, test_data, xAll_omics, pretrain_model, downstream_model, 
         name_embeddings, desc_embeddings, all_edge_index, internal_edge_index, 
         ppi_edge_index, device, checkpoint_path=None, run_timestamp=None):
    """Evaluation function for GRetriever model"""
    print("\n" + "="*50)
    print("Starting evaluation on test set...")
    print("="*50)
    
    # Load trained model if checkpoint path is provided
    if checkpoint_path is not None:
        print(f"Loading model from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        gretriever.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded successfully from epoch {checkpoint['epoch']+1} with loss {checkpoint['loss']:.16f}")
    
    # Create timestamp if not provided
    if run_timestamp is None:
        run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set model to evaluation mode
    gretriever.eval()
    
    # NER extraction setup
    api_key = "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"
    api_url = "https://api.openai.com/v1/chat/completions"
    
    # Tracking metrics
    test_metrics = []
    
    with torch.no_grad():
        for sample_id, sample_info in tqdm(test_data.items(), desc="Evaluating test samples"):
            print(f"\n{'='*80}")
            print(f"🧬 Evaluating on Sample ID: {sample_id}")
            print(f"{'='*80}\n")
            
            # Generate components for GRetriever input
            question, additional_text_context, _ = prepare_gretriever_input(sample_info)
            
            # Prepare input data
            selected_sample_index = sample_info["sample_dti_index"]
            selected_sample_x = xAll_omics[selected_sample_index, :].reshape(-1, args.omics_feature_dim)
            selected_sample_x_emb = pre_embed(pretrain_model, downstream_model, selected_sample_x, 
                                       name_embeddings, desc_embeddings, 
                                       all_edge_index, internal_edge_index, ppi_edge_index, device)

            selected_sample_x_emb = selected_sample_x_emb.to(device=gretriever.llm.device, dtype=torch.float32)
            sample_edge_index = all_edge_index.to(device=gretriever.llm.device)
            batch = torch.zeros(selected_sample_x.size(1), dtype=torch.long, device=gretriever.llm.device)
            
            # Optimize inference with reduced token generation and faster decoding
            response = gretriever.inference(
                question=[question], 
                x=selected_sample_x_emb,
                edge_index=all_edge_index.to(gretriever.llm.device), 
                batch=batch, 
                additional_text_context=[additional_text_context], 
                max_out_tokens=1000  # Reduced from 4000 for faster inference
            )
            
            # Clear GPU memory after each inference
            torch.cuda.empty_cache()
            
            # Process response
            initial_reasoning = response[0]
            # Truncate the initial reasoning after special tokens
            if '</s>' in initial_reasoning:
                initial_reasoning = initial_reasoning.split('</s>')[0]
            elif '[/s]' in initial_reasoning:
                initial_reasoning = initial_reasoning.split('</INST>')[0]
            elif '</p>' in initial_reasoning:
                initial_reasoning = initial_reasoning.split('</p>')[0]
            elif '[/p]' in initial_reasoning:
                initial_reasoning = initial_reasoning.split('[/p]')[0]
            elif '</INST>' in initial_reasoning:
                initial_reasoning = initial_reasoning.split('</INST>')[0]
            elif '[/INST]' in initial_reasoning:
                initial_reasoning = initial_reasoning.split('[/INST]')[0]
            print("Model Response:\n", initial_reasoning)
            
            # Extract proteins
            llm_response_hgnc_list = extract_proteins_with_chatgpt(initial_reasoning, api_key, api_url)
            print("\nExtracted Gene/Protein List:\n", llm_response_hgnc_list)
            
            # Get ground truth
            crispr_answer = sample_info["ground_truth_answer"]["top_bm_gene"]["hgnc_symbols"]
            crispr_answer_dict = sample_info["ground_truth_answer"]
            
            # Calculate metrics
            llm_metrics = calculate_metrics(llm_response_hgnc_list, crispr_answer)
            test_metrics.append(llm_metrics)
            
            # Build results and print metrics
            results_dict = build_and_print_results(
                initial_reasoning=initial_reasoning,
                llm_response_hgnc_list=llm_response_hgnc_list,
                llm_metrics=llm_metrics,
                crispr_answer=crispr_answer,
                crispr_answer_dict=crispr_answer_dict
            )
            
            # Save results
            save_results_to_json(
                sample_id=sample_id,
                results_dict=results_dict,
                output_dir=args.output_result_dir,
                timestamp=run_timestamp
            )
    
    return test_metrics

def main():
    args = arg_parse()
    # Set the device
    device = 'cpu' if args.device < 0 else (f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Make sure the device is set globally
    torch.cuda.set_device(args.device)
    # Set random seed for reproducibility
    seed_everything(args.seed)
    
    # Create output directories if they don't exist
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_result_dir).mkdir(parents=True, exist_ok=True)
    
    # Load KG omics data
    print("Loading Knowledge Graph data...")
    xAll_omics, omics_node_index_df, name_embeddings, desc_embeddings, all_edge_index, internal_edge_index, ppi_edge_index, ppi_edges = kg_data(args, device)
    # Get dimensions
    print(f"Knowledge Graph loaded: {xAll_omics.shape[1]} entities, {args.omics_feature_dim} features")
    # number of entity
    args.num_entity = xAll_omics.shape[1]

    # Load pretrain model
    pretrain_model = build_pretrain_model(args, device)
    pretrain_model.load_state_dict(torch.load(args.save_path))
    pretrain_model.eval()
    
    # Load model
    downstream_model = build_model(args, device)
    downstream_model.load_state_dict(torch.load(args.train_save_path))
    downstream_model.eval()

    # Load your finetuned LLM
    model_name = './Checkpoints/finetuned_model/CRISPR-QA-20250425_014540/checkpoint-125'
    base_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    
    # Initialize LLM
    print(f"Loading LLM from {model_name}")
    llm = LLM(model_name=model_name, base_model_name=base_model_name, num_params=8, dtype=torch.bfloat16)
    
    # Initialize GNN
    gnn = SimpleGNN(
        in_channels=args.omics_feature_dim,
        hidden_channels=8,
        out_channels=8
    ).to(llm.device)
    
    # Initialize GRetriever
    gretriever = GRetriever(
        llm=llm,
        gnn=gnn,
        use_lora=False,
        mlp_out_channels=4096,
        mlp_out_tokens=1
    )
    
    # Load training and testing data
    print(f"Loading training data from {args.train_json}")
    with open(args.train_json, "r") as f:
        train_data = json.load(f)
    
    print(f"Loading test data from {args.test_json}")
    with open(args.test_json, "r") as f:
        test_data = json.load(f)
    
    # Select operating mode (can be controlled by command-line argument)
    # mode = "eval"  # "train", "eval", or "both"
    mode = "eval"
    checkpoint_path = "./Checkpoints/gretriever/gretriever_checkpoint_epoch3.pt"
    
    run_timestamp = None
    
    if mode == "train" or mode == "both":
        # Train the model
        gretriever, run_timestamp = train(args, gretriever, train_data, xAll_omics, pretrain_model, 
                                          downstream_model, name_embeddings, desc_embeddings, 
                                          all_edge_index, internal_edge_index, ppi_edge_index, device)
    
    if mode == "eval" or mode == "both":
        # Evaluate the model
        test_metrics = eval(args, gretriever, test_data, xAll_omics, pretrain_model, 
                            downstream_model, name_embeddings, desc_embeddings, all_edge_index,
                            internal_edge_index, ppi_edge_index, device, 
                            checkpoint_path=checkpoint_path if mode == "eval" else None,
                            run_timestamp=run_timestamp)

if __name__ == "__main__":
    main()