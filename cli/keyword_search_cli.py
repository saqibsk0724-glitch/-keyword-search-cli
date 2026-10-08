import argparse
import json
import string

def matches(query , title , stopwords):
    translator = str.maketrans("" , "" , string.punctuation)
    query = query.lower().translate(translator)
    title = title.lower().translate(translator)


    query_tokens = [
        token for token in query.split()
        if token not in stopwords
    ]

    title_tokens = [
            token for token in title.split()
            if token not in stopwords
        ]

   

    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False



def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    with open("data/stopwords.txt" , "r") as f:
        stopwords = f.read().splitlines()

    translator = str.maketrans("" , "" , string.punctuation)

    stopwords = [
        word.lower() .translate(translator)
        for word in stopwords
    ]

    match args.command:
        case "search":
            with open("data/movies.json" , "r") as f:
                data = json.load(f)
            
            results = []
            for movie in data["movies"]:
                if matches(args.query , movie["title"] , stopwords):
                    results.append(movie)
                
            print(f"Searching for: {args.query}")
            for i , movie in enumerate(results[:5] , start=1):
                print(f"{i}. {movie['title']} ")
            pass
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()