import pickle
import uuid
import json
from pathlib import Path
from langchain.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever

# Initialize the vectorstore and retriever
def initialize_retriever(collection_name: str = "multi_modal_rag") -> MultiVectorRetriever:
    # The vectorstore to use to index the child chunks
    vectorstore = Chroma(collection_name=collection_name,
                         embedding_function=OpenAIEmbeddings(),
                         persist_directory="data/chroma_langchain_db"
                         )
    
    # The storage layer for the parent documents
    store = InMemoryStore()
    id_key = "doc_id"

    # The retriever (empty to start)
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )
    
    return retriever, id_key

# Method to add documents (texts/tables/images) to the retriever
def add_documents_to_retriever(retriever: MultiVectorRetriever, contents: list, summaries: list, id_key: str):
    doc_ids = [str(uuid.uuid4()) for _ in contents]
    summary_documents = [
        Document(page_content=summaries[i], metadata={id_key: doc_ids[i]})
        for i in range(len(summaries))
    ]
    retriever.vectorstore.add_documents(summary_documents)
    retriever.docstore.mset(list(zip(doc_ids, contents)))


def add_documents_to_retriever_image(retriever: MultiVectorRetriever, contents: list, summaries: list, id_key: str):
    doc_ids = [str(uuid.uuid4()) for _ in summaries]
    summary_documents = [
        Document(page_content=summaries[i], metadata={id_key: doc_ids[i]})
        for i in range(len(summaries))
    ]
    retriever.vectorstore.add_documents(summary_documents)
    retriever.docstore.mset(list(zip(doc_ids, contents)))

# Save retriever to file for later use
def save_retriever_and_docstore(retriever: MultiVectorRetriever, docstore_save_path: str):
    try:
        # Save Chroma vectorstore
        retriever.vectorstore.persist()  # This will save Chroma vectorstore to disk
        
        # Save InMemoryStore using pickle
        with open(docstore_save_path, 'wb') as f:
            pickle.dump(retriever.docstore, f)
        
        print(f"Docstore successfully saved to {docstore_save_path}")
    except Exception as e:
        raise Exception(f"Error saving retriever and docstore: {str(e)}")

# New method to handle full process of adding texts, tables, and images
def process_and_save_retriever(texts: list, tables: list, img_base64_list: list, table_summaries: list, image_summaries: list, save_path: str = "retriever_store"):
    try:
        # Initialize retriever
        retriever, id_key = initialize_retriever()


        # Add text documents to the retriever
        add_documents_to_retriever(retriever, texts, texts, id_key)

        # Add table summaries to the retriever
        add_documents_to_retriever(retriever, tables, table_summaries, id_key)

        # Extract summaries from image_summaries
        summary_img = [img['summary'] for img in image_summaries]  # Extract summaries from image_summaries

        # Add image summaries to the retriever
        add_documents_to_retriever_image(retriever, img_base64_list, summary_img, id_key)
        print(img_base64_list)
        save_retriever_and_docstore(retriever, "data/docstore.pkl")

        return retriever

    except Exception as e:
        print(f"An error occurred during processing: {str(e)}")

