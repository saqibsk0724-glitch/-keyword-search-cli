import os

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch

def normalize(scores : list[float]) -> list[float]:
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    if min_score == max_score:
        return [1.0] * len(scores)

    return [
        (score - min_score) / (max_score - min_score)
        for score in scores
    ]

def hybrid_score (bm25_score : float , semantic_score : float , alpha : float) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score



class HybridSearch:
    def __init__(self , documents : list[dict]) -> None:
        self.documents = documents

        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()

        if not os.path.exists(os.path.join("cache", "index.pkl")):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self , query : str , limit : int) -> list[dict]:
        self.idx.load()
        results = []

        for doc_id, score in self.idx.bm25_search(query, limit):
            document = self.idx.docmap[doc_id]
            results.append({
                "id": doc_id,
                "title": document["title"],
                "description": document["description"],
                "score": float(score),
            })

        return results


    def weighted_search(self , query : str , alpha : float , limit: int = 5) -> list[dict]:
        fetch_limit = 500 * limit

        keyword_results = self._bm25_search(query , fetch_limit)
        semantic_results = self.semantic_search.search_chunks(query , fetch_limit)

        keyword_normalized = normalize(
            [float(result["score"]) for result in keyword_results]
        )


        semantic_normalized = normalize(
            [float(result["score"]) for result in semantic_results]
        )

        combined = {}

        for result , score in zip(keyword_results , keyword_normalized):
            doc_id = result["id"]

            combined.setdefault(doc_id , {
                "document" : result,
                "keyword_score" : 0.0,
                "semantic_score" : 0.0
            })

            combined[doc_id]["document"] =  result
            combined[doc_id]["keyword_score"] = score

        for result , score in zip(semantic_results , semantic_normalized):
            doc_id = result["id"]

            combined.setdefault(doc_id , {
                "document" : result,
                "keyword_score" : 0.0,
                "semantic_score" : 0.0
            })

            combined[doc_id]["semantic_score"] = score
        results = []

        for doc_id , item in combined.items():
            keyword_score = item["keyword_score"]
            semantic_score = item["semantic_score"]
            doc = item["document"]

            results.append({
                "id" : doc_id,
                "title" : doc["title"],
                "document" : (
                    doc.get("document") or
                        doc.get("description" , "")
                ),
                "keyword_score" : keyword_score,
                "semantic_score": semantic_score,
                "hybrid_score" : hybrid_score(
                    keyword_score , semantic_score , alpha
                ),
                })

        results.sort(
            key=lambda result : result["hybrid_score"],
            reverse=True
        )
        return results[:limit]



    def rrf_search(self  , query : str , k : int , limit : int  = 10) -> list[dict]:
        raise NotImplementedError("RRF hybrid search not implemented yet")
