"""TwHIN relative-key attention on transformers 5, without loading pickle weights."""

import os
import re
import socket
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from twhin_bert_compat import install_twhin_relative_key_support  # noqa: E402
from transformers import BertConfig, BertModel  # noqa: E402
from transformers.models.bert.modeling_bert import BertSelfAttention  # noqa: E402


def _tiny_config():
    return BertConfig(
        hidden_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=16,
        vocab_size=32,
        max_position_embeddings=16,
        position_embedding_type="relative_key",
        attn_implementation="eager",
    )


def test_relative_key_module_uses_distance_weights_and_absolute_does_not():
    install_twhin_relative_key_support()
    relative = BertSelfAttention(_tiny_config(), layer_idx=0)
    absolute_config = _tiny_config()
    absolute_config.position_embedding_type = "absolute"
    absolute = BertSelfAttention(absolute_config, layer_idx=0)

    assert hasattr(relative, "distance_embedding")
    assert not hasattr(absolute, "distance_embedding")

    relative.eval()
    hidden = torch.ones(1, 3, 8)
    with torch.no_grad():
        first, _ = relative(hidden, attention_mask=None)
        center = relative.max_position_embeddings - 1
        relative.distance_embedding.weight[center].add_(5.0)
        second, _ = relative(hidden, attention_mask=None)
    assert first.shape == (1, 3, 8)
    assert not torch.equal(first, second)


def test_attention_scores_match_the_transformers_4_57_formula():
    """Oracle is the 4.57.3 equation, written out here rather than calling the patch."""
    install_twhin_relative_key_support()
    config = _tiny_config()
    layer = BertSelfAttention(config, layer_idx=0)
    layer.eval()
    torch.manual_seed(0)
    hidden = torch.randn(1, 4, config.hidden_size)
    with torch.no_grad():
        output, _ = layer(hidden, attention_mask=None)
        query = layer.query(hidden).view(1, 4, config.num_attention_heads, -1).transpose(1, 2)
        key = layer.key(hidden).view(1, 4, config.num_attention_heads, -1).transpose(1, 2)
        value = layer.value(hidden).view(1, 4, config.num_attention_heads, -1).transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-1, -2))
        positions_l = torch.arange(4).view(-1, 1)
        positions_r = torch.arange(4).view(1, -1)
        distance = positions_l - positions_r
        positional = torch.nn.functional.embedding(
            distance + config.max_position_embeddings - 1,
            layer.distance_embedding.weight,
        )
        relative = torch.einsum("bhld,lrd->bhlr", query, positional)
        scores = (scores + relative) * layer.scaling
        probs = torch.nn.functional.softmax(scores, dim=-1)
        expected = torch.matmul(probs, value).transpose(1, 2).contiguous().reshape(1, 4, config.hidden_size)
    assert torch.allclose(output, expected, atol=1e-6)


def test_cached_twhin_snapshot_loads_distance_weights_without_network():
    snapshot = os.environ.get("TWHIN_SNAPSHOT")
    if not snapshot:
        cached_root = Path(os.environ.get("HF_HOME", "")) / "hub" / "models--Twitter--twhin-bert-base"
        snapshots = list(cached_root.glob("snapshots/*")) if cached_root.is_dir() else []
        snapshot = str(snapshots[0]) if snapshots else ""
    if not snapshot or not Path(snapshot, "model.safetensors").is_file():
        pytest.skip("local Twitter/twhin-bert-base snapshot is not available")

    install_twhin_relative_key_support()
    from safetensors import safe_open
    from transformers import AutoModel

    original_connect = socket.socket.connect
    original_create = socket.create_connection

    def _blocked(*args, **kwargs):
        raise RuntimeError("hub access is not allowed for a cached snapshot")

    socket.create_connection = _blocked
    socket.socket.connect = lambda self, address: _blocked(address)
    try:
        model = AutoModel.from_pretrained(
            snapshot,
            revision="82ac392ce81f94560c391311ee2ddd024c5ac1fc",
            trust_remote_code=False,
            local_files_only=True,
            use_safetensors=True,
        )
    finally:
        socket.socket.connect = original_connect
        socket.create_connection = original_create
    model.eval()
    with safe_open(str(Path(snapshot) / "model.safetensors"), framework="pt", device="cpu") as handle:
        keys = [key for key in handle.keys() if key.endswith("distance_embedding.weight")]
        assert len(keys) == model.config.num_hidden_layers
        assert not any("pooler" in key for key in handle.keys())
        for key in keys:
            layer_index = int(re.search(r"layer\.(\d+)\.", key).group(1))
            expected = handle.get_tensor(key)
            actual = model.encoder.layer[layer_index].attention.self.distance_embedding.weight
            assert torch.equal(actual.detach().cpu(), expected)

    input_ids = torch.tensor([[2, 15, 80, 400, 3]])
    with torch.no_grad():
        faithful = model(input_ids).last_hidden_state
        for layer in model.encoder.layer:
            layer.attention.self.distance_embedding.weight.zero_()
        without_relative = model(input_ids).last_hidden_state
    assert faithful.shape[-1] == model.config.hidden_size
    assert not torch.allclose(faithful, without_relative)
