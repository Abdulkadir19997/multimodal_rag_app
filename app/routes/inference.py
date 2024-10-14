from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.ai_services.query_rag_chain import load_retriever_and_docstore, get_query_chain, get_result

router = APIRouter()

# Request body model
class QueryRequest(BaseModel):
    query: str

# Route to accept a query and return the processed content
@router.post("", summary="Get query answers from the preprocessed data")
async def get_query_answers(request: QueryRequest):
    try:
        # Load retriever and docstore
        retriever = load_retriever_and_docstore()
        if not retriever:
            raise HTTPException(status_code=500, detail="Failed to load retriever and docstore")

        # Get the query chain
        query_chain = get_query_chain(retriever)
        if not query_chain:
            raise HTTPException(status_code=500, detail="Failed to create query chain")

        # Process the query and get the result
        response_messages, docs_by_type = get_result(query_chain, retriever, request.query)


        if response_messages is None:
            raise HTTPException(status_code=500, detail="Failed to process the query")

        # Return the response and documents by type
        return {
            "response": response_messages,
            "docs_by_type": docs_by_type
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process the query: {str(e)}")
