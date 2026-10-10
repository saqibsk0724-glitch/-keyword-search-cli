import argparse

from lib.semantic_search import embed_text , verify_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("verify", help="Verify the embedding model")

    embed_parser = subparsers.add_parser(
        "embed_text",
        help="Generate an embedding for a text" ,
    )
    embed_parser.add_argument("text" , help="Text to embed")



    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case  "embed_text":
            embed_text(args.text)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
