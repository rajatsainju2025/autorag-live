import pickle

import numpy as np
import pytest

from autorag_live.cache.embedding_cache import EmbeddingCache


@pytest.mark.asyncio
async def test_save_async(tmp_path):
    cache = EmbeddingCache()
    cache.put("test", np.array([1, 2, 3]))

    filepath = tmp_path / "cache.pkl"
    await cache.save_async(str(filepath))

    assert filepath.exists()

    with open(filepath, "rb") as f:
        data = pickle.load(f)
        assert "cache" in data
        assert "timestamps" in data
        # Check if the key exists (md5 hash of "test")
        # We don't know the hash easily here without importing hashlib or using _make_key
        # But we can check if cache is not empty
        assert len(data["cache"]) == 1
