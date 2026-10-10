import argparse
import json 

from lib.semantic_search import (embed_text,
                                 verify_model, 
                                 verify_embeddings,
                                 embed_query_text,
                                 SemanticSearch
                                 )


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("verify", help="Verify the embedding model")

    subparsers.add_parser("verify_embeddings" , help="Verify movie embeddings")

    embed_parser = subparsers.add_parser(
        "embed_text",
        help="Generate an embedding for a text" ,
    )
    embed_parser.add_argument("text" , help="Text to embed")

    query_parser = subparsers.add_parser(
        "embed_query",
        help="Generate an embedding for a search query",
    )
    query_parser.add_argument("query" , help="Search query to embed")

    search_parser = subparsers.add_parser(
        "search",
         help="Search movies by meaning",
    )

    search_parser.add_argument("query" , help="Search query")
    search_parser.add_argument("--limit" , type=int , default=5 , help="Maximum number of results")



    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case  "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embed_query":
            embed_query_text(args.query)
        case "search":
            with open("data/movies.json" , "r" , encoding="utf-8") as file:
                data = json.load(file)

            documents = data["movies"]

            semantic_search = SemanticSearch()
            semantic_search.load_or_create_embeddings(documents)

            results = semantic_search.search(args.query , args.limit)

            for i , result in enumerate(results , start=1):
                print(
                    f"{i}. {result['title']}"
                    f"(score: {result['score']:.4f})"
                )
                print(f" {result['description']}")
                print()





        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
