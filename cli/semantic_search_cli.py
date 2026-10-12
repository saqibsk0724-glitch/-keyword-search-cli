import argparse
import json


from lib.semantic_search import (embed_text,
                                 verify_model,
                                 verify_embeddings,
                                 embed_query_text,
                                 SemanticSearch,
                                 semantic_chunk_text,
                                 ChunkedSemanticSearch
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

    chunk_parser = subparsers.add_parser(
        "chunk",
        help="Split text into fixed-size chunks"
    )
    chunk_parser.add_argument("text" , help="Text to chunk")
    chunk_parser.add_argument(
        "--chunk-size",
        type=int,
        default=200,
        help="Number of words per chunk"
    )
    chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Number of words shared between chunks"
    )

    search_chunked_parser = subparsers.add_parser(
        "search_chunked",
        help="Search movies using chunk embeddings"
    )

    search_chunked_parser.add_argument("query" , help="Search query")
    search_chunked_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of results"
    )


    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk",
        help="Split text into sentence-based chunks",
    )

    semantic_chunk_parser.add_argument(
        "text",
        help="Text to chunk"
    )

    semantic_chunk_parser.add_argument(
        "--max-chunk-size",
        type=int,
        default=4,
        help="Maximum sentences per chunk",
    )

    semantic_chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Number of sentences shared between chunks",
    )

    subparsers.add_parser(
        "embed_chunks",
        help="Generate embeddings for movie chunks"
    )


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

        case "chunk":
            words = args.text.split()
            chunk_size = args.chunk_size
            overlap = args.overlap

            if chunk_size <= 0:
                raise ValueError("chunk size must be positive")

            if overlap < 0 or overlap >= chunk_size:
                raise ValueError("Overlap must be non-negative and smaller than chunk size")

            chunks = []
            start = 0

            while start < len(words):
                chunk_words = words[start:start + chunk_size]
                chunks.append(" ".join(chunk_words))
                start += chunk_size - overlap


            print(f"Chunking {len(args.text)} characters")

            for i, chunk in enumerate(chunks , start=1):
                print(f"{i}. {chunk}")


        case "semantic_chunk":
            chunks = semantic_chunk_text(
                args.text,
                args.max_chunk_size,
                args.overlap,
            )

            print(f"Semantically chunking {len(args.text)} characters")

            for i, chunk in enumerate(chunks, start=1):
                print(f"{i}. {chunk}")


        case "embed_chunks":
            with open("data/movies.json" , "r" , encoding="utf-8") as file:
                data = json.load(file)

            documents = data["movies"]

            semantic_search = ChunkedSemanticSearch()
            embeddings = semantic_search.load_or_create_chunk_embeddings(documents)

            print(f"Generated {len(embeddings)} chunked embeddings")

        case "search_chunked":
            with open("data/movies.json" , "r" , encoding="utf-8") as file:
                data = json.load(file)

            documents = data["movies"]

            semantic_search = ChunkedSemanticSearch()
            semantic_search.load_or_create_chunk_embeddings(documents)

            results = semantic_search.search_chunks(args.query , args.limit)

            for i , result in enumerate(results , start=1):
                print(f"\n{i}. {result['title']} (score: {result['score']:.4f})")
                print(f"  {result['document']}...")




        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
