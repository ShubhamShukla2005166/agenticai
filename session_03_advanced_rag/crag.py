from __future__ import annotations

import json
import re
import sys
import textwrap
from pathlib import Path

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

GENERATOR_MODEL = "gpt-4o"
GRADER_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
TOP_K = 4

grader_llm = ChatOpenAI(model_name=GRADER_MODEL, temperature=0)
generator_llm = ChatOpenAI(model_name=GENERATOR_MODEL, temperature=0)

CORPUS_PATH= Path(__file__).resolve().parent / "data"/"lab1_crag_corpus.json"

def load_corpus() -> list[dict[str, str]]:
    if not CORPUS_PATH.exists():
        sys.exit(f"Corpus file not found at {CORPUS_PATH}. Please ensure the file exists.")
    entries = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    for entry in entries:
        entry["text"] = " ".join(entry["text"].split())
    return entries

CORPUS: list[dict[str, str]] = load_corpus()


def _normalize_for_match(text: str) -> set[str]:
    tokenized = re.sub(r"[^a-z0-9]+", " ", text.lower()).split()
    return set(tokenized)


class InMemoryVectorStore:
    def __init__(self, docs: list[Document]):
        self.docs = docs

    def similarity_search(self, query: str, k: int = TOP_K) -> list[Document]:
        if not self.docs:
            return []

        query_terms = _normalize_for_match(query)
        if not query_terms:
            return self.docs[:k]

        scored: list[tuple[float, Document]] = []
        for doc in self.docs:
            doc_terms = _normalize_for_match(doc.page_content)
            overlap = len(query_terms & doc_terms)
            score = overlap / max(len(query_terms), 1)
            scored.append((score, doc))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scored[:k]]


def build_vectorstore() -> InMemoryVectorStore:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ". "],
    )

    docs: list[Document] = []
    for entry in CORPUS:
        clean = textwrap.dedent(entry["text"]).strip()
        for chunk in splitter.split_text(clean):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={"doc_id": entry["id"], "title": entry["title"]},
                )
            )

    print(f"Indexed {len(docs)} document chunks from {len(CORPUS)} documents.")
    return InMemoryVectorStore(docs)

BASIC_RAG_PROMPT = """You are Meridian's internal policy assistant. Answer the following questions using the context below
Context: {context}
Question:{question}
Answer:"""

def basic_rag(store: InMemoryVectorStore, question: str) -> tuple[str, list[Document]]:
    print(f"Performing basic RAG for question: {question}")
    docs = store.similarity_search(question, k=TOP_K)
    context = "\n\n".join(
        f"[{doc.metadata['doc_id']} - {doc.metadata['title']}]\n{doc.page_content}" for doc in docs
    )
    prompt = BASIC_RAG_PROMPT.format(context=context, question=question)
    answer = generator_llm.invoke(prompt).content
    return answer, docs

def main():

    print("Building vector store from corpus...")
    store = build_vectorstore()
    print("Vector store built.")

    basic_answer, basic_docs = basic_rag(store, "What is Meridian's policy on remote work?")
    print(f"Answer: {basic_answer}")
    print(f"Sources: {[doc.metadata for doc in basic_docs]}")

if __name__ == "__main__":
    main()