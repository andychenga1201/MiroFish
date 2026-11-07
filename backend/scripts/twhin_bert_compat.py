"""Restore TwHIN-BERT relative-key attention on transformers 5.

``Twitter/twhin-bert-base`` sets ``position_embedding_type`` to ``relative_key``
and stores one ``distance_embedding`` per layer. transformers 4.57.3 applies
those weights inside ``BertSelfAttention``. transformers 5.22.2 removed that
path, so ``from_pretrained`` drops the tensors and the encoder runs as plain
BERT. The high transformers advisories that motivated staying on 5.x
(GHSA-29pf-2h5f-8g72, fixed in 5.3.0; GHSA-fgcw-684q-jj6r, fixed in 5.5.0)
are in other loading paths, not in this attention math.

The relative-score formula is the one in transformers 4.57.3
``BertSelfAttention.forward`` (Apache-2.0): add the query/distance term to
``QK^T`` before scaling by ``1/sqrt(head_size)``. This module does not change
the recommendation model, does not replace ``pooler_output`` with mean pooling,
and does not load ``pytorch_model.bin``. The published checkpoint has no
``pooler`` tensors; a freshly initialized pooler is the pre-existing behavior
of ``AutoModel`` on this masked-LM checkpoint.
"""

from __future__ import annotations

import os

_REPO_ID = "Twitter/twhin-bert-base"
# Snapshot verified in the offline safetensors load. Pinning avoids an unpinned Hub fetch.
_REVISION = "82ac392ce81f94560c391311ee2ddd024c5ac1fc"
_RELATIVE_TYPES = ("relative_key", "relative_key_query")
_SAFE_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
)
_INSTALLED = False
_ORIGINAL_INIT = None
_ORIGINAL_FORWARD = None
_ORIGINAL_EMB_INIT = None
_ORIGINAL_EMB_FORWARD = None


def install_twhin_relative_key_support() -> bool:
    """Patch BERT relative-key attention and the OASIS TwHIN loader.

    Returns True the first time the attention patch is installed.
    """
    global _INSTALLED, _ORIGINAL_INIT, _ORIGINAL_FORWARD
    from torch import nn
    from transformers.models.bert.modeling_bert import BertSelfAttention

    from transformers.models.bert.modeling_bert import BertEmbeddings

    if not getattr(BertSelfAttention, "_mirofish_relative_key", False):
        _ORIGINAL_INIT = BertSelfAttention.__init__
        _ORIGINAL_FORWARD = BertSelfAttention.forward
        _ORIGINAL_EMB_INIT = BertEmbeddings.__init__
        _ORIGINAL_EMB_FORWARD = BertEmbeddings.forward

        def _emb_init(self, config):
            _ORIGINAL_EMB_INIT(self, config)
            self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

        def _emb_forward(
            self,
            input_ids=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            past_key_values_length: int = 0,
        ):
            # transformers 4.57 adds absolute position vectors only for
            # position_embedding_type == "absolute". transformers 5 always adds
            # them, which changes TwHIN-BERT inputs before attention.
            if getattr(self, "position_embedding_type", "absolute") not in _RELATIVE_TYPES:
                return _ORIGINAL_EMB_FORWARD(
                    self,
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    inputs_embeds=inputs_embeds,
                    past_key_values_length=past_key_values_length,
                )
            return _relative_embeddings_forward(
                self,
                input_ids,
                token_type_ids,
                position_ids,
                inputs_embeds,
                past_key_values_length,
            )

        BertEmbeddings.__init__ = _emb_init
        BertEmbeddings.forward = _emb_forward

        def _init(self, config, is_causal: bool = False, layer_idx: int | None = None):
            _ORIGINAL_INIT(self, config, is_causal=is_causal, layer_idx=layer_idx)
            position_type = getattr(config, "position_embedding_type", "absolute")
            self.position_embedding_type = position_type
            if position_type in _RELATIVE_TYPES:
                self.max_position_embeddings = config.max_position_embeddings
                self.distance_embedding = nn.Embedding(
                    2 * config.max_position_embeddings - 1,
                    self.attention_head_size,
                )

        def _forward(
            self,
            hidden_states,
            attention_mask=None,
            past_key_values=None,
            **kwargs,
        ):
            position_type = getattr(self, "position_embedding_type", "absolute")
            if position_type not in _RELATIVE_TYPES or past_key_values is not None:
                return _ORIGINAL_FORWARD(
                    self,
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    **kwargs,
                )
            return _relative_key_forward(
                self,
                hidden_states,
                attention_mask,
                position_type,
            )

        BertSelfAttention.__init__ = _init
        BertSelfAttention.forward = _forward
        BertSelfAttention._mirofish_relative_key = True
        _INSTALLED = True

    _install_oasis_loader()
    return _INSTALLED


def local_twhin_snapshot(repo_id: str = _REPO_ID) -> str:
    """Return a local snapshot directory, downloading safetensors only if needed.

    A populated Hugging Face cache is used with no Hub call. The pickle
    ``pytorch_model.bin`` is not requested.
    """
    from huggingface_hub import snapshot_download, try_to_load_from_cache

    cached = try_to_load_from_cache(repo_id, "model.safetensors", revision=_REVISION)
    if isinstance(cached, str) and os.path.isfile(cached):
        return os.path.dirname(cached)
    return snapshot_download(
        repo_id,
        revision=_REVISION,
        allow_patterns=list(_SAFE_FILES),
    )


def _install_oasis_loader() -> None:
    try:
        import oasis.social_platform.recsys as recsys
    except Exception:
        return
    if getattr(recsys, "_mirofish_twhin_loader", False):
        return

    def get_twhin_tokenizer():
        if recsys.twhin_tokenizer is None:
            from transformers import AutoTokenizer

            recsys.twhin_tokenizer = AutoTokenizer.from_pretrained(
                local_twhin_snapshot(),
                revision=_REVISION,
                model_max_length=512,
                trust_remote_code=False,
                local_files_only=True,
            )
        return recsys.twhin_tokenizer

    def get_twhin_model(device):
        if recsys.twhin_model is None:
            from transformers import AutoModel

            model = AutoModel.from_pretrained(
                local_twhin_snapshot(),
                revision=_REVISION,
                trust_remote_code=False,
                local_files_only=True,
                use_safetensors=True,
            )
            model.eval()
            recsys.twhin_model = model.to(device)
        return recsys.twhin_model

    recsys.get_twhin_tokenizer = get_twhin_tokenizer
    recsys.get_twhin_model = get_twhin_model
    recsys._mirofish_twhin_loader = True


def _relative_embeddings_forward(
    self,
    input_ids,
    token_type_ids,
    position_ids,
    inputs_embeds,
    past_key_values_length: int,
):
    import torch

    if input_ids is not None:
        input_shape = input_ids.size()
    else:
        input_shape = inputs_embeds.size()[:-1]
    batch_size, seq_length = input_shape
    if position_ids is None:
        position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]
    if token_type_ids is None:
        if hasattr(self, "token_type_ids"):
            buffered_token_type_ids = self.token_type_ids.expand(position_ids.shape[0], -1)
            buffered_token_type_ids = torch.gather(buffered_token_type_ids, dim=1, index=position_ids)
            token_type_ids = buffered_token_type_ids.expand(batch_size, seq_length)
        else:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
    if inputs_embeds is None:
        inputs_embeds = self.word_embeddings(input_ids)
    embeddings = inputs_embeds + self.token_type_embeddings(token_type_ids)
    embeddings = self.LayerNorm(embeddings)
    embeddings = self.dropout(embeddings)
    return embeddings


def _relative_key_forward(self, hidden_states, attention_mask, position_type: str):
    """transformers 4.57.3 relative-key scores, then the transformers 5 scale-and-mask order."""
    import torch

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.attention_head_size)
    query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
    key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
    value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)

    query_length, key_length = query_layer.shape[2], key_layer.shape[2]
    position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
    position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
    distance = position_ids_l - position_ids_r
    positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
    positional_embedding = positional_embedding.to(dtype=query_layer.dtype)

    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    if position_type == "relative_key":
        relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
        attention_scores = attention_scores + relative_position_scores
    else:
        relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
        relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
        attention_scores = (
            attention_scores + relative_position_scores_query + relative_position_scores_key
        )

    attention_scores = attention_scores * self.scaling
    if attention_mask is not None:
        # transformers 5 gives a boolean keep-mask. transformers 4.57 adds
        # finfo.min on discarded positions and 0 on kept positions.
        if attention_mask.dtype == torch.bool:
            attention_scores = attention_scores.masked_fill(
                ~attention_mask,
                torch.finfo(attention_scores.dtype).min,
            )
        else:
            attention_scores = attention_scores + attention_mask
    attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
    attention_probs = self.dropout(attention_probs)
    context = torch.matmul(attention_probs, value_layer)
    context = context.transpose(1, 2).contiguous().reshape(*input_shape, -1)
    return context, attention_probs


def relative_key_scores(query_layer, key_layer, distance_weight, max_position_embeddings: int):
    """Public formula used by the equivalence test. ``query_layer`` is (batch, heads, seq, dim)."""
    import torch

    query_length, key_length = query_layer.shape[2], key_layer.shape[2]
    position_ids_l = torch.arange(query_length, dtype=torch.long, device=query_layer.device).view(-1, 1)
    position_ids_r = torch.arange(key_length, dtype=torch.long, device=query_layer.device).view(1, -1)
    distance = position_ids_l - position_ids_r
    positional_embedding = torch.nn.functional.embedding(
        distance + max_position_embeddings - 1,
        distance_weight,
    )
    return torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
