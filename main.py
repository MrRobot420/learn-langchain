import os
import tiktoken
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.helpers import llm, embeddings


async def stream_llm_response(prompt: str, temp=0):
    """Streams the response to the prompt from the LLM API asynchronously."""
    chunks = []
    async for chunk in llm(temperature=temp).astream(prompt):
        chunks.append(chunk)
        print(chunk.content, end="", flush=True)


def tokenizer(prompt: str):
    """Tokenizes the prompt using the OpenAI tokenizer."""
    encoding = tiktoken.encoding_for_model(os.environ.get("OPENAI_MODEL"))
    tokens = encoding.encode(prompt)
    decoded_tokens = [
        encoding.decode_single_token_bytes(token).decode("utf-8") for token in tokens
    ]
    for token in decoded_tokens:
        print(token)


def figure_out_dimensions(prompt: str):
    """Figure out the dimensions of the LLM model."""
    query_result = embeddings().embed_query(prompt)
    print(f"Dimensions of the prompt: {query_result[:3]}")


def split_documents_for(file_path: str):
    """Retrieves the top 5 sentences from the document using BM25."""
    document_loader = PyPDFLoader(file_path)
    documents = document_loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
    splitted_documents = splitter.split_documents(documents)

    return splitted_documents


def bm25_retriever(prompt: str, documents: list):
    """Retrieves the top 5 sentences from the document using BM25."""
    retriever = BM25Retriever.from_documents(documents=documents, k=3)
    top_sentences = retriever.get_relevant_documents(prompt)
    return top_sentences


if __name__ == "__main__":
    user_input = "How do I use the Flipper Zero to hack WiFi?"
    # tokenizer(user_input)
    figure_out_dimensions(user_input)
    split_documents = split_documents_for("src/data/FlipperZeroManual.pdf")
    top_sentences = bm25_retriever(user_input, split_documents)
    print(top_sentences)
