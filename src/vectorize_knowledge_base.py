"""
vectorize_knowledge_base.py

Converts a given knowledge base (JSON format) into vector embeddings, 
creates a FAISS index for efficient similarity search, and generates metadata.

Usage:
    python vectorize_knowledge_base.py --json_path "path/to/knowledge.json" \
                                       --output_index "knowledge_base.index" \
                                       --output_metadata "knowledge_base_metadata.json"
"""

import argparse
import json
import numpy as np
import faiss
import logging
from langchain.embeddings.openai import OpenAIEmbeddings

def paper_to_text(paper: dict) -> str:
    """
    Convert a paper dictionary into a single string for vectorization.

    Args:
        paper: A dictionary containing paper details.

    Returns:
        str: A formatted string containing the paper's title, link, and significance.
    """
    title = paper.get("title", "")
    link = paper.get("link", "")
    significance = paper.get("biggest_significance", "")
    text = f"Title: {title}\nLink: {link}\nSignificance: {significance}"
    return text

def load_papers(json_path: str) -> list:
    """
    Load papers from a JSON file and return a list of paper dictionaries.

    Args:
        json_path: Path to the JSON file.

    Returns:
        List: A list of dictionaries representing papers.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    papers = data.get("papers", [])
    return papers

def vectorize_texts(texts: list, embeddings_model: OpenAIEmbeddings) -> np.ndarray:
    """
    Convert a list of text descriptions into vector embeddings.

    Args:
        texts: List of text descriptions to vectorize.
        embeddings_model: OpenAI embedding model instance.

    Returns:
        np.ndarray: A NumPy array containing vector representations.
    """
    vectors = [embeddings_model.embed_query(text) for text in texts]
    return np.array(vectors, dtype="float32")

def build_faiss_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build and return a FAISS index for fast similarity search.

    Args:
        vectors: A NumPy array containing vector embeddings.

    Returns:
        faiss: A FAISS index for similarity search.
    """
    # Normalize vectors
    faiss.normalize_L2(vectors)
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index

def main():
    """
    Main function to handle argument parsing, processing, vectorization, and file saving.
    """
    # Set up logging configuration.
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(
        description="Convert a JSON file of papers into a FAISS vector database with metadata."
    )
    parser.add_argument("--json_path", type=str, required=True,
                        help="Path to the JSON file containing the papers.")
    parser.add_argument("--output_index", type=str, default="knowledge_base.index",
                        help="Output file path for the FAISS index.")
    parser.add_argument("--output_metadata", type=str, default="knowledge_base_metadata.json",
                        help="Output file path for the metadata JSON file.")
    args = parser.parse_args()
    
    logging.info(f"Loading papers from {args.json_path}...")
    papers = load_papers(args.json_path)
    if not papers:
        logging.error("No papers found in the JSON file.")
        return
    
    logging.info(f"Loaded {len(papers)} papers.")
    
    # Convert each paper into a text string for vectorization
    texts = [paper_to_text(paper) for paper in papers]
    
    # Initialize the embeddings model 
    logging.info("Initializing OpenAI embeddings model...")
    embeddings_model = OpenAIEmbeddings()
    
    logging.info("Vectorizing papers...")
    vectors = vectorize_texts(texts, embeddings_model)
    
    logging.info("Building FAISS index...")
    index = build_faiss_index(vectors)
    logging.info(f"FAISS index built with {index.ntotal} vectors.")
    
    logging.info(f"Saving index to {args.output_index}...")
    faiss.write_index(index, args.output_index)
    logging.info("Index saved successfully.")
    
    logging.info(f"Saving metadata to {args.output_metadata}...")
    with open(args.output_metadata, "w", encoding="utf-8") as f:
        json.dump(papers, f, indent=2)
    logging.info("Metadata saved successfully.")

if __name__ == "__main__":
    main()
