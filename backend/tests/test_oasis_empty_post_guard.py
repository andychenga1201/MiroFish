"""camel-oasis 0.2.5 TWHIN refresh must not IndexError on an empty post table.

These tests load the installed 0.2.5 Platform method. Heavy imports (torch,
camel, transformers) are stubbed and network connects are denied. No LLM,
embedding, or recommendation-model call is part of the empty-table path.
"""

import asyncio
import importlib
import importlib.metadata
import logging
import os
import socket
import sqlite3
import sys
import tempfile
import types
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from oasis_compat import install_empty_post_table_guard  # noqa: E402

_INSERTED_MODULES: list[str] = []


def _ensure_module(name: str) -> types.ModuleType:
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    try:
        imported = importlib.import_module(name)
    except Exception:
        imported = None
    if imported is not None:
        return imported
    module = types.ModuleType(name)
    sys.modules[name] = module
    _INSERTED_MODULES.append(name)
    return module


def _install_heavy_stubs() -> None:
    sklearn = _ensure_module("sklearn")
    feature_extraction = _ensure_module("sklearn.feature_extraction")
    text = _ensure_module("sklearn.feature_extraction.text")

    class TfidfVectorizer:
        def __init__(self, *args, **kwargs):
            pass

    text.TfidfVectorizer = TfidfVectorizer
    feature_extraction.text = text
    sklearn.feature_extraction = feature_extraction

    metrics = _ensure_module("sklearn.metrics")
    pairwise = _ensure_module("sklearn.metrics.pairwise")
    pairwise.cosine_similarity = lambda *args, **kwargs: None
    metrics.pairwise = pairwise
    sklearn.metrics = metrics

    torch = _ensure_module("torch")
    torch.device = lambda name="cpu": name
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.no_grad = lambda: (lambda fn: fn)
    torch.cat = lambda *args, **kwargs: None
    torch.tensor = lambda *args, **kwargs: None

    sentence_transformers = _ensure_module("sentence_transformers")

    class SentenceTransformer:
        def __init__(self, *args, **kwargs):
            pass

    sentence_transformers.SentenceTransformer = SentenceTransformer

    camel = _ensure_module("camel")
    camel.__path__ = getattr(camel, "__path__", [])
    embeddings = _ensure_module("camel.embeddings")

    class OpenAIEmbedding:
        def __init__(self, *args, **kwargs):
            raise AssertionError("OpenAI embeddings are not available in offline tests")

    embeddings.OpenAIEmbedding = OpenAIEmbedding
    camel.embeddings = embeddings
    camel_types = _ensure_module("camel.types")

    class EmbeddingModelType:
        TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"

    camel_types.EmbeddingModelType = EmbeddingModelType
    camel.types = camel_types

    transformers = _ensure_module("transformers")

    class AutoModel:
        pass

    class AutoTokenizer:
        pass

    transformers.AutoModel = AutoModel
    transformers.AutoTokenizer = AutoTokenizer
    _ensure_module("numpy")


def _load_platform_module():
    distribution = importlib.metadata.distribution("camel-oasis")
    if distribution.version != "0.2.5":
        pytest.fail(
            f"expected camel-oasis 0.2.5, found {distribution.version}"
        )
    oasis_dir = Path(distribution.locate_file("oasis"))
    platform_file = oasis_dir / "social_platform" / "platform.py"
    if not platform_file.is_file():
        pytest.fail(f"camel-oasis platform.py is missing at {platform_file}")

    _install_heavy_stubs()
    oasis = _ensure_module("oasis")
    oasis.__path__ = [str(oasis_dir)]
    oasis.__package__ = "oasis"
    oasis.__version__ = distribution.version

    # Import the submodule directly so oasis/__init__.py does not pull camel,
    # igraph, or neo4j. The namespace entry above keeps that __init__ from running.
    import oasis.social_platform.platform as platform_module

    return platform_module


@pytest.fixture(scope="module")
def oasis_platform():
    work = Path(tempfile.mkdtemp(prefix="oasis-empty-post-"))
    previous = os.getcwd()
    os.chdir(work)
    try:
        platform_module = _load_platform_module()
    finally:
        os.chdir(previous)
    platform_cls = platform_module.Platform
    original = platform_cls.update_rec_table
    forbidden = {"calls": 0}

    def forbid_twhin(*args, **kwargs):
        forbidden["calls"] += 1
        raise AssertionError("TWHIN refresh must not run in the empty-table test")

    platform_module.rec_sys_personalized_twh = forbid_twhin
    try:
        yield platform_cls, original, platform_module, forbidden
    finally:
        for name in list(sys.modules):
            if name == "oasis" or name.startswith("oasis."):
                sys.modules.pop(name, None)
        for name in reversed(_INSERTED_MODULES):
            sys.modules.pop(name, None)
        _INSERTED_MODULES.clear()


def _deny_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("offline test blocked a network call")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


def _capture_logger():
    records = []

    class _Handler(logging.Handler):
        def emit(self, record):
            records.append((record.levelname, record.getMessage()))

    handler = _Handler()
    logger = logging.getLogger("social.twitter")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return records, handler, logger


def _new_platform(platform_cls, db_path: Path, recsys_type: str = "twhin-bert"):
    return platform_cls(db_path=str(db_path), recsys_type=recsys_type)


def test_unpatched_twhin_refresh_logs_empty_post_index_error(tmp_path, oasis_platform, monkeypatch):
    platform_cls, original, _module, forbidden = oasis_platform
    _deny_network(monkeypatch)
    platform = _new_platform(platform_cls, tmp_path / "empty.db")
    platform.db.execute("INSERT INTO rec (user_id, post_id) VALUES (1, 1)")
    platform.db.commit()
    records, handler, logger = _capture_logger()
    try:
        asyncio.run(original(platform))
        messages = records
    finally:
        logger.removeHandler(handler)
        platform.db.close()

    assert forbidden["calls"] == 0
    assert ("INFO", "Starting to refresh recommendation system cache...") in messages
    assert ("ERROR", "list index out of range") in messages
    assert not any(level == "ERROR" and "network" in message.lower() for level, message in messages)


def test_guard_skips_empty_twhin_table_without_error(tmp_path, oasis_platform, monkeypatch):
    platform_cls, _original, _module, forbidden = oasis_platform
    installed = install_empty_post_table_guard(platform_cls)
    assert install_empty_post_table_guard(platform_cls) is False
    assert installed or getattr(platform_cls.update_rec_table, "_mirofish_empty_post_guard", False)
    _deny_network(monkeypatch)

    platform = _new_platform(platform_cls, tmp_path / "guarded-empty.db")
    platform.db.execute("INSERT INTO rec (user_id, post_id) VALUES (1, 1)")
    platform.db.commit()
    records, handler, logger = _capture_logger()
    try:
        result = asyncio.run(platform.update_rec_table())
        remaining = platform.db.execute("SELECT COUNT(*) FROM rec").fetchone()[0]
        messages = list(records)
    finally:
        logger.removeHandler(handler)
        platform.db.close()

    assert result is None
    assert remaining == 1
    assert forbidden["calls"] == 0
    assert ("ERROR", "list index out of range") not in messages
    assert not any(level == "ERROR" for level, _message in messages)
    assert any(
        "Skipping recommendation cache refresh because the post table is empty." in message
        for _level, message in messages
    )
    assert not any(
        message == "Starting to refresh recommendation system cache..."
        for _level, message in messages
    )


def test_guard_still_refreshes_twhin_when_posts_exist(tmp_path, oasis_platform, monkeypatch):
    platform_cls, _original, platform_module, _forbidden = oasis_platform
    install_empty_post_table_guard(platform_cls)
    calls = []

    def record_twhin(*args, **kwargs):
        calls.append((args, kwargs))
        return []

    previous = platform_module.rec_sys_personalized_twh
    platform_module.rec_sys_personalized_twh = record_twhin
    _deny_network(monkeypatch)
    platform = _new_platform(platform_cls, tmp_path / "with-post.db")
    platform.db.execute(
        "INSERT INTO post (user_id, content, created_at) VALUES (?, ?, ?)",
        (1, "seed", 1),
    )
    platform.db.commit()
    records, handler, logger = _capture_logger()
    try:
        asyncio.run(platform.update_rec_table())
        messages = list(records)
    finally:
        logger.removeHandler(handler)
        platform.db.close()
        platform_module.rec_sys_personalized_twh = previous

    assert len(calls) == 1
    _args, kwargs = calls[0]
    assert kwargs.get("use_openai_embedding") is False
    assert ("INFO", "Starting to refresh recommendation system cache...") in messages
    assert ("ERROR", "list index out of range") not in messages


def test_guard_leaves_non_twhin_refresh_unchanged():
    class RedditPlatform:
        def __init__(self):
            self.recsys_type = "reddit"
            self.calls = 0
            connection = sqlite3.connect(":memory:")
            connection.execute("CREATE TABLE post (post_id INTEGER)")
            self._connection = connection
            self.db_cursor = connection.cursor()

        async def update_rec_table(self):
            self.calls += 1
            post_table = []
            _latest = post_table[-1]

    assert install_empty_post_table_guard(RedditPlatform) is True
    platform = RedditPlatform()
    with pytest.raises(IndexError):
        asyncio.run(platform.update_rec_table())
    assert platform.calls == 1


def test_guard_ignores_methods_that_do_not_index_the_last_post():
    class AlreadySafe:
        async def update_rec_table(self):
            return "safe"

    original = AlreadySafe.update_rec_table
    assert install_empty_post_table_guard(AlreadySafe) is False
    assert AlreadySafe.update_rec_table is original


def test_simulation_scripts_install_the_guard():
    for name in (
        "run_twitter_simulation.py",
        "run_reddit_simulation.py",
        "run_parallel_simulation.py",
    ):
        source = (SCRIPTS / name).read_text(encoding="utf-8")
        assert "install_empty_post_table_guard()" in source
