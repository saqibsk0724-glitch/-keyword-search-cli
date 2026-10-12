import json
import os
import numpy as np
import re
from sentence_transformers import SentenceTransformer


class SemanticSearch:

    def __init__(self , model_name : str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name , device="cpu")
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self , text):
        if not text or not text.strip():
            raise ValueError("Text cannot be empty or whitespace")
        embedding = self.model.encode([text])[0]
        return embedding


    def build_embeddings(self , documents):
        self.documents = documents
        self.document_map = {
            doc['id']: doc for doc in documents
        }

        texts = [
            f"{doc['title']}: {doc['description']}"
            for doc in documents
        ]

        self.embeddings = self.model.encode(
            texts,
            show_progress_bar = True,
        )

        os.makedirs("cache" , exist_ok=True)
        np.save("cache/movie_embeddings.npy" , self.embeddings)

        return self.embeddings

    def load_or_create_embeddings(self , documents):
        self.documents = documents
        self.document_map = {
            doc["id"]: doc for doc in documents
        }
        embeddings_path = "cache/movie_embeddings.npy"

        if os.path.exists(embeddings_path):
            self.embeddings = np.load(embeddings_path)

            if len(self.embeddings) == len(documents):
                return self.embeddings

        return self.build_embeddings(documents)

    def search(self , query , limit):
        if self.embeddings is None:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )

        query_embedding = self.generate_embedding(query)

        scored_documents = []

        for i , document in enumerate(self.documents):
            score = cosine_similarity(
                query_embedding,
                self.embeddings[i],
            )
            scored_documents.append((score , document))

        scored_documents.sort(key=lambda item : item[0], reverse=True)

        results = []
        for score , document in scored_documents[:limit]:
            results.append({
                "score" : float(score),
                "title" : document["title"],
                "description" : document["description"],
            })
        return results


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self , model_name : str = "all-MiniLM-L6-v2")-> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self , documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {
            doc["id"]: doc for doc in documents
        }

        all_chunks = []
        chunk_metadata = []

        for movie_idx , document in enumerate(self.documents):
            description = document["description"]

            if not description or not description.strip():
                continue

            chunks = semantic_chunk_text(description , max_chunk_size=4 , overlap=1)

            total_chunks = len(chunks)

            for chunk_idx , chunk in enumerate(chunks):
                all_chunks.append(chunk)

                chunk_metadata.append({
                    "movie_idx" : movie_idx,
                    "chunk_idx" : chunk_idx,
                    "total_chunks" : total_chunks,
                })

        self.chunk_embeddings = self.model.encode(all_chunks , show_progress_bar=True)


        self.chunk_metadata = chunk_metadata

        os.makedirs("cache" , exist_ok=True)

        np.save("cache/chunk_embeddings.npy" , self.chunk_embeddings)


        with open( "cache/chunk_metadata.json" , "w" , encoding="utf-8") as f:
            json.dump(
                {
                    "chunks" : chunk_metadata,
                    "total_chunks" : len(all_chunks),
                },
                f,
                indent=2
            )
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self , documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}

        embeddings_path = "cache/chunk_embeddings.npy"
        metadata_path = "cache/chunk_metadata.json"

        if(os.path.exists(embeddings_path) and os.path.exists(metadata_path)):
            self.chunk_embeddings = np.load(embeddings_path)
            with open(metadata_path , "r" , encoding="utf-8") as f:
                metadata = json.load(f)

            self.chunk_metadata = metadata["chunks"]

            return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)



    def search_chunks(self , query : str , limit : int = 10) -> list[dict]:
        query_embedding =  self.generate_embedding(query)
        chunk_scores = []



        for i , embedding in enumerate(self.chunk_embeddings):
            score = cosine_similarity(embedding ,  query_embedding)

            chunk_scores.append({
                "chunk_idx" : self.chunk_metadata[i]["chunk_idx"],
                "movie_idx" : self.chunk_metadata[i]["movie_idx"],
                "score" : score,
            })

        movie_scores = {}

        for chunk in chunk_scores:
            movie_idx = chunk["movie_idx"]
            score = chunk["score"]

            if movie_idx not in movie_scores or score > movie_scores[movie_idx]:
                movie_scores[movie_idx] = score


        sorted_movies = sorted(movie_scores.items() , key=lambda item : item[1] , reverse=True)[:limit]

        results = []

        for movie_idx , score in sorted_movies:
            doc = self.documents[movie_idx]

            results.append({
                "id" : doc["id"],
                "title" : doc["title"],
                "document" : doc["description"][:100],
                "score" : round(score , 3)

            })
        return results









def verify_model():
    semantic_search = SemanticSearch()
    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")


def embed_text(text):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings():
    with open("data/movies.json" , "r" , encoding="utf-8") as file:
        data = json.load(file)
    documents = data["movies"]
    semantic_search = SemanticSearch()
    embeddings = semantic_search.load_or_create_embeddings(documents)
    print(f"Number of docs : {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")


def embed_query_text(query):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(query)

    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")


def cosine_similarity(vec1 : np.ndarray , vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1 , vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

def semantic_chunk_text(text : str , max_chunk_size : int = 4 , overlap: int = 1) -> list[str]:
    if max_chunk_size <= 0:
        raise ValueError("Max chunk size must be positive")

    if overlap < 0 or overlap >= max_chunk_size:
        raise ValueError("Overlap must be non-negative and smaller than max chunk size")

    text = text.strip()


    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+" , text.strip())

    if len(sentences) == 1 and not sentences[0].endswith(("." , "!" , "?")):
        sentences = [text]

    sentences = [sentence.strip() for sentence in sentences]
    chunks = []
    start = 0
    while start < len(sentences):
        chunk_sentences = sentences[start : start + max_chunk_size]
        chunk = " ".join(sentence for sentence in chunk_sentences if sentence)

        if chunk.strip():
            chunks.append(chunk)

        if start + max_chunk_size >= len(sentences):
            break
        start += max_chunk_size - overlap

    return chunks
