import argparse
import json

from lib.hybrid_search import HybridSearch

def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    if min_score == max_score:
        return [1.0] * len(scores)

    return [
        (score - min_score) / (max_score - min_score) for score in scores
    ]



def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command" , help="Available commands")

    



    normalize_parser = subparsers.add_parser(
        "normalize",
        help="Normalize a list of scores"
    )

    normalize_parser.add_argument(
        "scores",
        nargs="*",
        type=float,
        help="Scores to normalize"
    )

    weighted_parser = subparsers.add_parser(
        "weighted-search",
        help="Search using weighted keyword and semantic scores"
    )

    weighted_parser.add_argument("query" , help="Search_query")
    weighted_parser.add_argument("--alpha" , type=float , default=0.5)
    weighted_parser.add_argument("--limit" , type=int , default=5)

    rrf_parser = subparsers.add_parser(
        "rrf-search",
        help="Search using Reciprocal Rank Fusion"
    )

    rrf_parser.add_argument("query" , help="search query")
    rrf_parser.add_argument("-k" , type=int , default=60)
    rrf_parser.add_argument("--limit" , type=int , default=5)



    args = parser.parse_args()

    match args.command:
        

        case "normalize":
            scores = normalize_scores(args.scores)

            for score in scores:
                print(f"* {score:.4f}")

        case "weighted-search":
            with open("data/movies.json" , "r" , encoding="utf-8") as file:
                documents = json.load(file)["movies"]

            hybrid = HybridSearch(documents)

            results = hybrid.weighted_search(
                args.query,
                args.alpha, 
                args.limit
            )

            for i , result in enumerate(results[:args.limit] , start=1):
                print(f"{i}.{result['title']}")
                print(f" Hybrid Score: {result['hybrid_score']:.3f}")
                print(
                    f" BM25: {result['keyword_score']:.3f}, "
                    f"Semantic: {result['semantic_score']:.3f}"
                )
                print(f" {result['document'][:100]}...")

        case "rrf-search":
            with open("data/movies.json", "r", encoding="utf-8") as file:
                documents = json.load(file)["movies"]

            hybrid = HybridSearch(documents)
            results = hybrid.rrf_search(
                args.query,
                args.k,
                args.limit,
        )

            for i, result in enumerate(results[:args.limit], start=1):
                print(f"{i}. {result['title']}")
                print(f"  RRF Score: {result['rrf_score']:.3f}")

                bm25_rank = result["bm25_rank"]
                semantic_rank = result["semantic_rank"]

                print(
                    f"  BM25 Rank: {bm25_rank}, "
                    f"Semantic Rank: {semantic_rank}"
                )
                print(f"  {result['document'][:100]}...")
                print()



        case _:
            parser.print_help()



if __name__ == "__main__":
    main()






        
        