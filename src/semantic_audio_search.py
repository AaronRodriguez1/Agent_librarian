"""
semantic_audio_search.py

A chatbot that searches FAISS indexes built from transcript data and a knowledge base.
It retrieves and ranks relevant text based on a user query.

Usage:
    python semantic_audio_search.py --transcript_index "path/to/transcript.index" \
                      --transcript_metadata "path/to/transcription_metadata.json" \
                      --kb_index "path/to/knowledge_base.index" \
                      --kb_metadata "path/to/knowledge_base_metadata.json" \
                      --top_k 3
"""

import argparse
import json
import os
import numpy as np
import faiss
from langchain.embeddings.openai import OpenAIEmbeddings

def load_index(path: str) -> faiss.Index:
    """
    Load a FAISS index from a file.

    Args:
        path: Path to the FAISS index file.

    Returns:
        faiss: The loaded FAISS index.
    """
    return faiss.read_index(path)

def load_metadata(path: str) -> list:
    """
    Load metadata from a JSON file.

    Args:
        path: Path to the JSON metadata file.

    Returns:
        List: List of metadata entries corresponding to FAISS index entries.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def search_index(query: str, model: OpenAIEmbeddings, index: faiss.Index, metadata: list, top_k: int):
    """
    Search a FAISS index for the most relevant results based on the query.

    Args:
        query: The user query to search for.
        model: The embeddings model used for vectorizing the query.
        index: The FAISS index to search.
        metadata: The corresponding metadata entries.
        top_k: The number of top results to retrieve.

    Returns:
        List: A list of top matching results with scores and associated metadata.
    """
    q_vec = model.embed_query(query)
    q_vec = np.array(q_vec, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(q_vec)
    
    distances, indices = index.search(q_vec, top_k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(metadata):
            results.append({"score": float(dist), "data": metadata[idx]})
    return results

def main():
    """
    Main function to initialize and run the chatbot.
    It loads FAISS indexes, metadata, and performs query searches interactively.
    """
    parser = argparse.ArgumentParser(description="A simple chatbot that searches transcript and KB indexes.")
    parser.add_argument("--transcript_index", type=str, required=True,
                        help="Path to the FAISS index for the transcript.")
    parser.add_argument("--transcript_metadata", type=str, required=True,
                        help="Path to the transcript metadata JSON file.")
    parser.add_argument("--kb_index", type=str, required=True,
                        help="Path to the FAISS index for the knowledge base.")
    parser.add_argument("--kb_metadata", type=str, required=True,
                        help="Path to the KB metadata JSON file.")
    parser.add_argument("--top_k", type=int, default=3,
                        help="Number of top results to return from each index.")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Please set the OPENAI_API_KEY environment variable.")

    # Load indexes and metadata
    transcript_idx = load_index(args.transcript_index)
    kb_idx = load_index(args.kb_index)
    transcript_meta = load_metadata(args.transcript_metadata)
    kb_meta = load_metadata(args.kb_metadata)

    # Initialize the embeddings model
    model = OpenAIEmbeddings()

    print("\nChatbot (type 'exit' to quit)")
    while True:
        query = input("\nUser: ")
        if query.lower() in ["exit", "quit"]:
            print("Exiting.")
            break

        # Search both indexes
        trans_results = search_index(query, model, transcript_idx, transcript_meta, args.top_k)
        kb_results = search_index(query, model, kb_idx, kb_meta, args.top_k)

        # Print transcript results
        print("\nTranscript Results:")
        for i, res in enumerate(trans_results, start=1):
            data = res["data"]
            timestamp = data.get("timestamp", "N/A")
            text = data.get("text", "")
            print(f"{i}. Score: {res['score']:.3f} | Timestamp: {timestamp}")
            print(f"   Text: {text[:150]}...\n")

        # Print knowledge base results
        print("Knowledge Base Results:")
        for i, res in enumerate(kb_results, start=1):
            data = res["data"]
            title = data.get("title", "N/A")
            link = data.get("link", "N/A")
            significance = data.get("biggest_significance", "")
            print(f"{i}. Score: {res['score']:.3f} | Title: {title}")
            print(f"   Link: {link}")
            print(f"   Significance: {significance[:150]}...\n")

if __name__ == "__main__":
    main()
