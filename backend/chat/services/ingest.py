import argparse
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import BSHTMLLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# constants
THIS_DIR = Path(__file__).resolve().parent
CHROMA_DB_PATH = str(THIS_DIR.parent.parent / "knowledge" / "chroma_db")


def load_file_to_documents(filepath: str) -> list[Document]:
    """
    Loads an HTML file and chunks it into a list of LangChain Document objects.
    """
    # load html documents and strip html tags
    loader = BSHTMLLoader(filepath)
    documents: list[Document] = loader.load()

    # split documents into chunks for better semantic search
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks: list[Document] = splitter.split_documents(documents)

    return chunks


def store_embeddings(docs: list[Document], embeddings: HuggingFaceEmbeddings) -> Chroma:
    vectorstore = Chroma.from_documents(
        documents=docs, embedding=embeddings, persist_directory=CHROMA_DB_PATH
    )
    return vectorstore


def ingest_files(filepaths: list[str]):
    print("Creating embeddings model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True},
    )
    for filepath in filepaths:
        print(f"Embedding file {filepath}...")
        docs: list[Document] = load_file_to_documents(filepath)
        store_embeddings(docs, embeddings)
    print("Done storing requested files in Chroma database.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest files into Chroma vector database"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dir", type=str, help="Directory containing files to ingest")
    group.add_argument("-i", nargs="+", type=str, help="List of file paths to ingest")

    args = parser.parse_args()

    if args.dir:
        dir_path = Path(args.dir)
        if not dir_path.is_dir():
            print(f"Error: {args.dir} is not a valid directory")
            exit(1)
        filepaths = [str(f) for f in dir_path.iterdir() if f.is_file()]
    else:
        filepaths = args.i

    print(f"Ingesting files: {filepaths}")
    ingest_files(filepaths)
