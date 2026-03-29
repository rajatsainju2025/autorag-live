from autorag_live.chunking.chunker import SentenceChunker


def test_sentence_chunker_preserves_overlap_and_bounds():
    chunker = SentenceChunker(max_chunk_size=30, min_chunk_size=1, overlap_sentences=1)

    chunks = chunker.chunk("Alpha beta. Gamma delta. Epsilon zeta.")

    assert [chunk.text for chunk in chunks] == [
        "Alpha beta. Gamma delta.",
        "Gamma delta. Epsilon zeta.",
    ]
    assert chunks[0].metadata.start_char == 0
    assert chunks[0].metadata.end_char == len(chunks[0].text)
    assert chunks[1].metadata.start_char == len("Alpha beta. ")
    assert chunks[1].metadata.end_char == chunks[1].metadata.start_char + len(chunks[1].text)


def test_sentence_chunker_keeps_small_final_chunk_when_first_chunk_absent():
    chunker = SentenceChunker(max_chunk_size=100, min_chunk_size=50, overlap_sentences=0)

    chunks = chunker.chunk("Short sentence.")

    assert len(chunks) == 1
    assert chunks[0].text == "Short sentence."
