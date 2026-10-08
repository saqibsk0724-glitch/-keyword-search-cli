import argparse
import json
import string
import os
import pickle
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()


def load_movies():
    with open("data/movies.json" , "r") as f:
        data = json.load(f)

    return data["movies"]

def tokenize_text(text):
    translator = str.maketrans("" , "" , string.punctuation)
    text = text.lower().translate(translator)

    with open("data/stopwords.txt" , "r") as f:
        stopwords = f.read().splitlines()
    stopwords = [
        word.lower().translate(translator)
        for word in stopwords
    ]

    tokens = [
        stemmer.stem(token)
        for token in text.split()
        if token not in stopwords
    ]
    return tokens


class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {}

    def __add_document(self , doc_id , text):
        tokens = tokenize_text(text)

        for token in tokens:
            if token not in self.index:
                self.index[token] = set()

            self.index[token].add(doc_id)

    def get_documents(self , term):
            return sorted(self.index.get(term , set()))

    def build(self):
        movies = load_movies()

        for m in movies:
            self.docmap[m["id"]] = m
        
            text = f"{m['title']} {m['description']}"
            self.__add_document(m["id"] , text)   

    def save(self):
        os.makedirs("cache" , exist_ok=True)

        with open("cache/index.pkl" , "wb") as f:
            pickle.dump(self.index , f)

        with open("cache/docmap.pkl" , "wb") as f:
            pickle.dump(self.docmap , f)
            
def matches(query , title , stopwords , stemmer):
    translator = str.maketrans("" , "" , string.punctuation)
    query = query.lower().translate(translator)
    title = title.lower().translate(translator)


    query_tokens = [
        stemmer.stem(token)
        for token in query.split()
        if token not in stopwords
    ]

    title_tokens = [
            stemmer.stem(token)
             for token in title.split()
            if token not in stopwords
        ]

   

    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False

def build_command():
    index = InvertedIndex()
    index.build()
    index.save()
    docs = index.get_documents("merida")
    print(f"First document for token 'merida' ={docs[0]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build" , help="Build the inverted text")

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
                if matches(args.query , movie["title"] , stopwords , stemmer):
                    results.append(movie)
                
            print(f"Searching for: {args.query}")
            for i , movie in enumerate(results[:5] , start=1):
                print(f"{i}. {movie['title']} ")
            pass

        case "build":
            build_command()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()