import argparse
import json
import string


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            with open("data/movies.json" , "r") as f:
                data = json.load(f)
            translator = str.maketrans("" , "" , string.punctuation)
            query = args.query.lower().translate(translator)
            results = []
            for movie in data["movies"]:
                title = movie["title"].lower().translate(translator)
                if query in title:
                    results.append(movie)
            print(f"Searching for: {args.query}")
            for i , movie in enumerate(results[:5] , start=1):
                print(f"{i}. {movie['title']} ")
            pass
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()