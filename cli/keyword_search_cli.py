import argparse
import json
import string
import os
import pickle
import math
from nltk.stem import PorterStemmer
from collections import Counter
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
        self.term_frequencies = {}

    def __add_document(self , doc_id , text):
        tokens = tokenize_text(text)

        self.term_frequencies[doc_id] = Counter()

        for token in tokens:
            if token not in self.index:
                self.index[token] = set()

            self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token] += 1

    def get_tf(self , doc_id , term):
        return self.term_frequencies[doc_id].get(term , 0)

    def get_document(self , term):
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
        
        with open("cache/term_frequencies.pkl" , "wb") as f:
            pickle.dump(self.term_frequencies , f)

    def load(self):
        with open("cache/index.pkl" , "rb") as f:
            self.index = pickle.load(f)
        with open("cache/docmap.pkl" , "rb") as f:
            self.docmap = pickle.load(f)
        with open("cache/term_frequencies.pkl" , "rb") as f:
            self.term_frequencies = pickle.load(f)

def tokenize_term(term):
    tokens = tokenize_text(term)
    if len(tokens) != 1:
        raise Exception("Term must contain exactly one token")
    return tokens[0]




            
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
   


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build" , help="Build the inverted text")


    tf_parser = subparsers.add_parser("tf" , help="Get term frequency")
    tf_parser.add_argument("doc_id" , type=int)
    tf_parser.add_argument("term" , type=str)

    idf_parser = subparsers.add_parser("idf" , help="Get inverse document frequency")
    idf_parser.add_argument("term" , type=str)

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
            index = InvertedIndex()

            try:
                index.load()
            except FileNotFoundError:
                print("Index not found. Please run the build command first.")
                return

           
            
           
            query_tokens = tokenize_text(args.query)
           
            results = []

            for token in query_tokens:
                docs = index.get_document(token)

                for doc_id in docs:
                    if doc_id not in results:
                        results.append(doc_id)
                    if len(results) == 5:
                       break
                if(len(results) == 5):
                    break

            for doc_id in results:
                movie = index.docmap[doc_id]    
                print(f"{movie['title']} ({doc_id})")

        case "idf":
            index = InvertedIndex()
            index.load()

            term = tokenize_term(args.term)
            df = len(index.get_document(term))
            idf = math.log(len(index.docmap) / (df+1))

            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")


        case "tf":
            index = InvertedIndex()
            try:
                index.load()
            except FileNotFoundError:
                print("Index not found. Please run the build command first")
                return 
            term = tokenize_term(args.term)
            frequency = index.get_tf(args.doc_id , term)

            print(frequency)

        case "build":
            build_command()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()