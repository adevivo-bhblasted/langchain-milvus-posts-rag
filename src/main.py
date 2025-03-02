import argparse

from .milvus_utils.vectorstore import add_document_from_text

from .query import query_posts


def main():
    parser = argparse.ArgumentParser(description='RAG System CLI')
    parser.add_argument('--action', type=str, choices=['create', 'query'], 
                       required=True, help='Mode: create documents or query database')
    parser.add_argument('--input', type=str, help='Input text or query')
    
    args = parser.parse_args()

    if args.action == 'create':
        if not args.input:
            args.input = input("Enter your text to ingest: ")
        add_document_from_text(args.input)
    elif args.action == 'query':
        if not args.input:
            args.input = input("Enter your query: ")
        query_posts(args.input)
    
    print("Done!")

if __name__ == "__main__":
    main()