import pickle
import chromadb
from langchain.vectorstores import Chroma
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.embeddings import OpenAIEmbeddings
from langchain.storage import InMemoryStore
from base64 import b64decode
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.schema.messages import HumanMessage

# Load docstore from disk
def load_docstore(docstore_save_path: str) -> InMemoryStore:
    try:
        # Load InMemoryStore using pickle
        with open(docstore_save_path, 'rb') as f:
            docstore = pickle.load(f)
        print(f"Docstore successfully loaded from {docstore_save_path}")
        return docstore
    except Exception as e:
        raise Exception(f"Error loading docstore: {str(e)}")
    

def load_retriever_and_docstore(collection_name: str = "multi_modal_rag", persist_directory: str = "data/chroma_langchain_db", docstore_save_path: str = "data/docstore.pkl") -> MultiVectorRetriever:
    try:
        # Load Chroma vectorstore
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=OpenAIEmbeddings(),
            persist_directory=persist_directory
        )

        # Load docstore from pickle file
        docstore = load_docstore(docstore_save_path)

        # Initialize the retriever with the loaded vectorstore and docstore
        retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            docstore=docstore,
            id_key="doc_id"
        )

        print("Retriever and docstore loaded successfully")
        return retriever

    except Exception as e:
        print(f"An error occurred while loading the retriever and docstore: {str(e)}")
        return None


def split_image_text_types(docs):
    ''' Split base64-encoded images and texts '''
    b64 = []
    text = []
    for doc in docs:
        try:
            b64decode(doc)
            b64.append(doc)
        except Exception as e:
            text.append(doc)
    return {
        "images": b64,
        "texts": text
        }


def prompt_func(dict):
    format_texts = "\n".join(dict["context"]["texts"])
    return [
        HumanMessage(
            content=[
                {"type": "text", "text": f"""Answer the question based only on the following context, which can include text, tables, and the below image, tell us which table or figure or section, and depending on that asnwer the question:
            Question: {dict["question"]}

            Text and tables:
            {format_texts}
            """}
                        ]
                    )
                ]


def get_query_chain(loaded_retriever):
    try:
        model = ChatOpenAI(temperature=0, model="gpt-4o", max_tokens=1024)
        

        # RAG pipeline
        chain = (
            {"context": loaded_retriever | RunnableLambda(split_image_text_types), "question": RunnablePassthrough()}
            | RunnableLambda(prompt_func)
            | model
            | StrOutputParser()
        )

        return chain
    
    except Exception as e:
        print(f"An error occurred while creating query chain {str(e)}")
        return None


def get_result(chain, loaded_retriever, query):
    try:

        docs = loaded_retriever.get_relevant_documents(query)  
        docs_by_type = split_image_text_types(docs)      
        response = chain.invoke(query)        

#         response = chain.invoke(
#     "What is the change in wild fires from 1993 to 2022?"
# )
        return response, docs_by_type
    except Exception as e:
        print(f"An error occurred while creating query chain {str(e)}")
        return None



# Example usage of loading the retriever
# def main():
#     # Load the retriever from the saved Chroma DB in retriever_store
#     loaded_retriever = load_retriever_and_docstore()

#     # Test the loaded retriever by asking a query
#     if loaded_retriever:
#         query = "What is the change in wildfires from 1993 to 2022?"
#         docs = loaded_retriever.get_relevant_documents(query)
#         print(docs)
#         # Print the retrieved documents

#     docs_by_type = split_image_text_types(docs)
#     docs_by_type["images"][0]
# if __name__ == "__main__":
#     main()