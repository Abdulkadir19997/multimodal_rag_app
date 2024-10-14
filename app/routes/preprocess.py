from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
from app.ai_services.create_image_summaries import get_image_summaries
from app.ai_services.create_table_summaries import get_table_summaries
from app.ai_services.preprocess_vector_db import process_and_save_retriever
from app.ai_services.render_preprocess_pdf import process_pdf

from config import settings
import shutil
import os
import tempfile

router = APIRouter()

# Router to accept the PDF file and return the processed content
@router.post("", summary="Process a PDF file and extract categorized content")
async def process_pdf_route(pdf_file: UploadFile = File(...)):
    try:
        # Create a temporary directory to store the uploaded file
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = os.path.join(temp_dir, pdf_file.filename)
            
            # Save the uploaded PDF file to the temporary directory
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(pdf_file.file, buffer)
            
            # Process the saved PDF file
            tables, texts = process_pdf(temp_file_path)
            text_summaries, table_summaries = get_table_summaries(texts, tables)
            image_summaries, img_base64_list = get_image_summaries()
            
            # Process and save the retriever
            retriever = process_and_save_retriever(texts, tables, img_base64_list, table_summaries, image_summaries)
            
            # Return the processed result (tables, texts, image summaries)
            return {
                "message": f"{pdf_file.filename} processed successfully"
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process the PDF: {str(e)}")
