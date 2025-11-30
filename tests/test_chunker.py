from api.state import Document
from tools.chunker import TextChunker


def test_chunker_creates_overlapping_chunks():
    text = " ".join([f"word{i}" for i in range(1500)])
    doc = Document(url="http://example.com", title="Example", content=text)
    chunker = TextChunker(chunk_size=200, overlap=50)
    chunks = chunker.chunk_document(doc)
    assert chunks
    # Expect multiple overlapping chunks
    assert len(chunks) > 1
    assert all("chunk_index" in c.metadata for c in chunks)
