import os
from fastapi import FastAPI, APIRouter
import openai
import uvicorn
from app.routes.preprocess import router as preprocessor
from app.routes.inference import router as inference
from config import settings

# langsmith traces
openai.api_key = settings.OPENAI_API_KEY
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"]=settings.LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"]="langchain-multimodal-papers-pdf"

app = FastAPI(title="Multimodal RAG API")

router = APIRouter(prefix="/app/demo")

# Middleware'ı ekleyin
# app.middleware("http")(catch_exceptions)
router.include_router(preprocessor, prefix="/preprocess", tags=["Render and preprocess pdf data to vectorDB"])
router.include_router(inference, prefix="/inference", tags=["Query on by doing semantic search on similar data in pdf"])
# # dependencies=[Depends(JWTBearer())]  authentication for api add

app.include_router(router)
# Run the application using Uvicorn
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=5004, reload=True)
