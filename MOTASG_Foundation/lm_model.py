# embed the disease name for bmg_disease_df by bert-based model
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Dict
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset

class SentenceDataset(Dataset):
    def __init__(self, sentences: List[str]):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]


class TextEncoder():
    def __init__(self, model_path: str = "dmis-lab/biobert-v1.1", device: str = "cuda"):
        """
        Args:
            model_path (str, optional): Path to the deberta model. Defaults to 'dmis-lab/biobert-v1.1'.
            device (str, optional): Device to run the model on ('cpu' or 'cuda'). Defaults to 'cpu'.
        """
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """
        Load the deberta model and tokenizer from the specified model path.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModel.from_pretrained(self.model_path).to(self.device)

    def generate_embeddings(self, sentences: List[str], batch_size: int = 32, text_emb_dim: int = 64) -> torch.Tensor:
        """
        Generate a single-dimensional embedding for each sentence.

        Args:
            sentences (List[str]): List of sentences to embed.
            batch_size (int, optional): Batch size for DataLoader. Defaults to 32.

        Returns:
            List[float]: List of single-dimensional embeddings.
        """
        dataset = SentenceDataset(sentences)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        embedding_batches = []
        for batch in tqdm(dataloader, desc="Embedding sentences", unit="batch"):
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            # Handle single batch case properly
            mean_embeddings = torch.mean(outputs.last_hidden_state, dim=1)  # [batch_size, hidden_dim]
            
            # For adaptive pooling, we need to reshape for 1D adaptive pooling
            # [batch_size, 1, hidden_dim] -> [batch_size, 1, text_emb_dim] -> [batch_size, text_emb_dim]
            batch_size = mean_embeddings.size(0)
            reshaped = mean_embeddings.view(batch_size, 1, -1)
            projected = torch.nn.functional.adaptive_avg_pool1d(reshaped, output_size=text_emb_dim)
            projected = projected.squeeze(1)  # Only squeeze dimension 1, keep batch dimension
            embedding_batches.append(projected)
        return torch.cat(embedding_batches, dim=0)

    def save_embeddings(self, embeddings, output_npy_path):
        """
        Save embeddings to a .npy file.
        
        Args:
            embeddings (torch.Tensor): The embeddings to save.
            output_npy_path (str): Path to save the embeddings file.
        """
        # Move embeddings to CPU before converting to numpy
        embeddings_cpu = embeddings.cpu().numpy()
        np.save(output_npy_path, embeddings_cpu)
        print(f"Embeddings saved at {output_npy_path} with shape {embeddings_cpu.shape}")