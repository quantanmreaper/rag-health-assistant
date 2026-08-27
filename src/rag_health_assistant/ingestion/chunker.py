"""
Chunker for splitting medical guideline text into semantic chunks with context preservation.
"""

from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ..config import CHUNK_SIZE, CHUNK_OVERLAP


def create_text_splitter() -> RecursiveCharacterTextSplitter:
    """
    Creates a recursive character text splitter tuned for clinical and guideline text.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n\n",
            "\n## ",
            "\n### ",
            "\n\n",
            "\n- ",
            "\n• ",
            "\n",
            ". ",
            "; ",
            " ",
            ""
        ],
        length_function=len,
        is_separator_regex=False
    )


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Splits documents into smaller semantic chunks and adds chunk-level metadata.
    """
    splitter = create_text_splitter()
    chunked_docs: List[Document] = []

    for doc_idx, doc in enumerate(documents):
        splits = splitter.split_text(doc.page_content)
        for split_idx, split_text in enumerate(splits):
            if len(split_text.strip()) < 30:
                continue  # Skip very short fragments/page artifacts
            
            chunk_metadata = dict(doc.metadata)
            chunk_metadata["chunk_id"] = f"{doc.metadata.get('source', 'doc')}_p{doc.metadata.get('page', 1)}_c{split_idx}"
            chunk_metadata["chunk_index"] = split_idx
            
            chunked_docs.append(
                Document(
                    page_content=split_text.strip(),
                    metadata=chunk_metadata
                )
            )

    print(f"Split {len(documents)} document pages into {len(chunked_docs)} semantic chunks.")
    return chunked_docs
