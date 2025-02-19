"""
vectorize_transcript.py

This script reads a transcript JSON file, vectorizes the transcript using OpenAI embeddings, 
builds a FAISS index for fast similarity search, and saves both the index and metadata.

Usage:
    python vectorize_transcript.py --transcript_json "path/to/transcription.json" \
                                   --output_index "transcription.index"
"""

import argparse
import json
import os
import numpy as np
import faiss
import logging
from langchain.embeddings.openai import OpenAIEmbeddings
from typing import List

def load_transcript(transcript_path: str) -> List[str]:
    """
    Load transcript JSON and return a list of text segments.

    Args:
        transcript_path: Path to the transcript JSON file.

    Returns:
        List: A list of transcript text segments.
    """
    with open(transcript_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return [entry["text"] for entry in data.get("audio_data", [])]

def vectorize_texts(texts: list, embeddings_model: OpenAIEmbeddings) -> np.ndarray:
    """
    Convert a list of text segments into vector embeddings.

    Args:
        texts: List of transcript segments to vectorize.
        embeddings_model: OpenAI embedding model instance.

    Returns:
        np.ndarray: A NumPy array containing vector representations of text segments.
    """
    vectors = [embeddings_model.embed_query(text) for text in texts]
    return np.array(vectors, dtype="float32")

def build_faiss_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build a FAISS index for fast similarity search.

    Args:
        vectors: A NumPy array containing vector embeddings.

    Returns:
        faiss: A FAISS index object for similarity search.
    """
    faiss.normalize_L2(vectors)
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index

def main():
    """
    Main function to process the transcript, vectorize text, and build FAISS index.
    """
    # Set up logging configuration.
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(
        description="Vectorize a transcript JSON file and build a FAISS index."
    )
    parser.add_argument("--transcript_json", type=str, required=True,
                        help="Path to the transcript JSON file.")
    parser.add_argument("--output_index", type=str, default="transcript.index",
                        help="File path to save the FAISS index (default: transcript.index).")
    args = parser.parse_args()

    # Ensure the OpenAI API key is set in your environment.
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Please set the OPENAI_API_KEY environment variable.")

    # Load the transcript.
    logging.info("Loading transcript...")
    texts = load_transcript(args.transcript_json)
    if not texts:
        logging.error("No transcript text found in the JSON file.")
        return
    logging.info(f"Found {len(texts)} transcript segments.")

    # Initialize the embeddings model.
    logging.info("Initializing OpenAI embeddings model...")
    embeddings_model = OpenAIEmbeddings()

    # Vectorize the transcript segments.
    logging.info("Vectorizing transcript segments...")
    vectors = vectorize_texts(texts, embeddings_model)

    # Build the FAISS index.
    logging.info("Building FAISS index...")
    index = build_faiss_index(vectors)
    logging.info(f"FAISS index built with {index.ntotal} vectors.")

    # Save the FAISS index.
    logging.info(f"Saving FAISS index to {args.output_index}...")
    faiss.write_index(index, args.output_index)

    logging.info("Vectorization complete!")

if __name__ == "__main__":
    main()
