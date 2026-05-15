import hashlib
import os
from typing import List, Optional

import torch
import torch.nn as nn


class ESMEncoder(nn.Module):
    """
    Sequence encoder with two execution paths:
    1. official ESM-2 embeddings when `fair-esm` is installed and enabled;
    2. a lightweight Transformer fallback so the project remains runnable.
    """

    def __init__(
        self,
        model_name,
        output_dim,
        max_length=1024,
        num_layers=2,
        num_heads=8,
        ffn_dim=512,
        dropout=0.1,
        use_pretrained=False,
        freeze_pretrained=True,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.output_dim = output_dim
        self.max_length = max_length
        self.use_pretrained = False
        self.freeze_pretrained = freeze_pretrained
        self.cache_dir = cache_dir

        self.esm_model = None
        self.alphabet = None
        self.batch_converter = None
        self.output_proj = None
        self.esm_layer_index = None

        if use_pretrained:
            try:
                import esm  # type: ignore

                loader = getattr(esm.pretrained, model_name, None)
                if loader is None:
                    raise AttributeError(f"Unknown ESM checkpoint: {model_name}")
                self.esm_model, self.alphabet = loader()
                self.batch_converter = self.alphabet.get_batch_converter()
                self.esm_layer_index = self.esm_model.num_layers
                self.output_proj = nn.Linear(self.esm_model.embed_dim, output_dim)
                self.use_pretrained = True

                if freeze_pretrained:
                    self.esm_model.eval()
                    for parameter in self.esm_model.parameters():
                        parameter.requires_grad = False

                if self.cache_dir:
                    os.makedirs(self.cache_dir, exist_ok=True)
            except Exception:
                self.use_pretrained = False

        if not self.use_pretrained:
            self.token_embed = nn.Embedding(32, output_dim, padding_idx=0)
            self.pos_embed = nn.Embedding(max_length + 2, output_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=output_dim,
                nhead=num_heads,
                dim_feedforward=ffn_dim,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(output_dim)

    def _cache_path(self, sequence: str) -> Optional[str]:
        if not self.cache_dir:
            return None
        digest = hashlib.sha1(f"{self.model_name}:{sequence}".encode("utf-8")).hexdigest()
        return os.path.join(self.cache_dir, f"{digest}.pt")

    def _encode_with_pretrained(self, sequences: List[str], device: torch.device) -> torch.Tensor:
        cached_embeddings: List[Optional[torch.Tensor]] = [None] * len(sequences)
        uncached = []

        for index, sequence in enumerate(sequences):
            cache_path = self._cache_path(sequence)
            if cache_path and os.path.exists(cache_path):
                cached_embeddings[index] = torch.load(cache_path, map_location="cpu")
            else:
                uncached.append((index, sequence))

        if uncached:
            batch = [(str(index), sequence[: self.max_length]) for index, sequence in uncached]
            _, _, tokens = self.batch_converter(batch)
            tokens = tokens.to(device)

            forward_context = torch.no_grad() if self.freeze_pretrained else torch.enable_grad()
            with forward_context:
                result = self.esm_model(tokens, repr_layers=[self.esm_layer_index], return_contacts=False)
            representations = result["representations"][self.esm_layer_index]

            for offset, (index, sequence) in enumerate(uncached):
                length = min(len(sequence), self.max_length)
                embedding = representations[offset, 1 : length + 1].detach().cpu()
                cached_embeddings[index] = embedding
                cache_path = self._cache_path(sequence)
                if cache_path:
                    torch.save(embedding, cache_path)

        batch_size = len(sequences)
        max_len = max(min(len(sequence), self.max_length) for sequence in sequences)
        hidden = torch.zeros(batch_size, max_len, self.esm_model.embed_dim, device=device)
        for index, embedding in enumerate(cached_embeddings):
            if embedding is None:
                continue
            hidden[index, : embedding.size(0)] = embedding.to(device)
        return hidden

    def forward(self, token_ids, padding_mask=None, sequences: Optional[List[str]] = None):
        if self.use_pretrained:
            if sequences is None:
                raise ValueError("Sequences are required when using the pretrained ESM encoder.")
            hidden_states = self._encode_with_pretrained(sequences, token_ids.device)
            return self.norm(self.output_proj(hidden_states))

        batch_size, seq_len = token_ids.shape
        position_ids = torch.arange(seq_len, device=token_ids.device).unsqueeze(0).expand(batch_size, -1)
        hidden_states = self.token_embed(token_ids) + self.pos_embed(position_ids)
        encoded = self.encoder(hidden_states, src_key_padding_mask=padding_mask)
        return self.norm(encoded)
