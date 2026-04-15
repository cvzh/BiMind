import torch
from torch.utils.data import Dataset


class NewsDataset(Dataset):
    """Dataset for the custom-transformer (non-LLM) BiMind model."""

    def __init__(self, sequences, pos_feats, content_features, knowledge_features, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.pos_feats = torch.tensor(pos_feats, dtype=torch.float)      # [N, L, POS_DIM]
        self.content_features = torch.tensor(content_features, dtype=torch.float)
        self.knowledge_features = torch.tensor(knowledge_features, dtype=torch.float)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.sequences[idx],
            self.pos_feats[idx],
            self.content_features[idx],
            self.knowledge_features[idx],
            self.labels[idx],
        )


class LLMNewsDataset(Dataset):
    """Dataset for the LLM-backbone BiMind model with pre-tokenised inputs."""

    def __init__(
        self,
        input_ids,
        attention_masks,
        pos_feats,
        content_features,
        knowledge_features,
        labels,
    ):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.pos_feats = pos_feats
        self.content_features = torch.tensor(content_features, dtype=torch.float)
        self.knowledge_features = torch.tensor(knowledge_features, dtype=torch.float)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.input_ids[idx],
            self.attention_masks[idx],
            self.pos_feats[idx],
            self.content_features[idx],
            self.knowledge_features[idx],
            self.labels[idx],
        )
