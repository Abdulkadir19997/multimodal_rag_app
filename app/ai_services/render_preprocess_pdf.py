from fastapi import HTTPException
from unstructured.partition.pdf import partition_pdf
from config import settings
import shutil
import os


# Method to process and categorize the PDF file
def process_pdf(file_path: str):
    try:
        # Extract images, tables, and chunk text
        raw_pdf_elements = partition_pdf(
            filename=file_path,
            extract_images_in_pdf=True,
            infer_table_structure=True,
            chunking_strategy="by_title",
            max_characters=4000,
            new_after_n_chars=3800,
            combine_text_under_n_chars=2000,
            image_output_dir_path=settings.FIGURES_PATH,
        )
        
        # Categorize by type
        tables = []
        texts = []
        for element in raw_pdf_elements:
            if "unstructured.documents.elements.Table" in str(type(element)):
                tables.append(str(element))
            elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
                texts.append(str(element))
        
        return tables, texts
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")