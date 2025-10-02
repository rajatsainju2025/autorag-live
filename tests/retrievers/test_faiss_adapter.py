

from autorag_live.retrievers.faiss_adapter import (
    DenseRetriever,
    SentenceTransformerRetriever,
    create_dense_retriever,
    load_retriever_index,
    save_retriever_index,
)


def test_dense_retriever_fallback():
    """Test dense retriever with deterministic fallback."""
    retriever = DenseRetriever()

    documents = ["The sky is blue.", "The sun is bright.", "Foxes are mammals."]
    retriever.build_index(documents)

    results = retriever.search("bright sky", k=2)
    assert len(results) == 2
    assert all(isinstance(score, float) for _, score in results)
    assert all(doc in documents for doc, _ in results)


def test_sentence_transformer_retriever():
    """Test sentence transformer retriever (may fall back to deterministic)."""
    retriever = SentenceTransformerRetriever()

    documents = ["The sky is blue.", "The sun is bright.", "Foxes are mammals."]
    retriever.build_index(documents)

    results = retriever.search("bright sky", k=2)
    assert len(results) == 2
    assert all(isinstance(score, float) for _, score in results)


def test_create_dense_retriever():
    """Test factory function."""
    retriever = create_dense_retriever("sentence-transformer")
    assert isinstance(retriever, SentenceTransformerRetriever)


def test_save_load_retriever(tmp_path):
    """Test saving and loading retriever index."""
    retriever = DenseRetriever()
    documents = ["The sky is blue.", "The sun is bright."]
    retriever.build_index(documents)

    # Save
    save_retriever_index(retriever, str(tmp_path))

    # Load
    loaded_retriever = load_retriever_index(str(tmp_path))

    assert loaded_retriever.documents == documents
    assert loaded_retriever.model_name == retriever.model_name

    # Test search on loaded retriever
    results = loaded_retriever.search("blue sky", k=1)
    assert len(results) == 1
