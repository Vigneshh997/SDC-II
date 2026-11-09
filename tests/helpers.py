"""Artificial fixtures for software tests only. Never clinical evidence."""

from types import SimpleNamespace

import torch
from torch import nn
from torchvision import transforms

from backend.models.multimodal import MultimodalModel


class Tokenizer:
    all_special_ids = [0, 1, 2]

    def __call__(self, notes, return_tensors=None, **kwargs):
        ids = [[1] + [3 + sum(map(ord, word)) % 29 for word in note.split()][:254] + [2] for note in notes]
        width = max(map(len, ids))
        masks = [[1] * len(row) + [0] * (width - len(row)) for row in ids]
        ids = [row + [0] * (width - len(row)) for row in ids]
        result = {"input_ids": ids, "attention_mask": masks}
        return {k: torch.tensor(v) for k, v in result.items()} if return_tensors else result

    def convert_ids_to_tokens(self, ids):
        return [str(i) for i in ids]


class TinyText(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8)
        self.embedding = nn.Embedding(32, 8)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, **kwargs):
        value = self.embedding(input_ids) if inputs_embeds is None else inputs_embeds
        return SimpleNamespace(last_hidden_state=value)


class TinyTrunk(nn.Module):
    num_features = 8
    num_prefix_tokens = 1

    def __init__(self):
        super().__init__()
        self.patch = nn.Conv2d(3, 8, 2, stride=2)

    def forward_features(self, images):
        patch = self.patch(images).flatten(2).transpose(1, 2)
        return torch.cat([patch.mean(1, keepdim=True), patch], 1)


def tiny_model(**kwargs):
    vision = nn.Module()
    vision.trunk = TinyTrunk()
    transform = transforms.Compose([transforms.Resize((8, 8)), transforms.ToTensor()])
    kwargs.update(d_model=16, nhead=4, num_layers=2)
    kwargs.pop("local_files_only", None)
    model = MultimodalModel(encoders=(vision, TinyText(), Tokenizer(), transform, transform), **kwargs)
    model.metadata_transform.fit([{"age": 20, "sex": "female"}, {"age": 60, "sex": "male"}])
    return model


def inputs():
    return (
        torch.randn(2, 3, 8, 8),
        ["software fixture one", "software fixture two words"],
        torch.tensor([[20, 1, 0, 0, 0], [60, 0, 1, 0, 0]], dtype=torch.float32),
    )
