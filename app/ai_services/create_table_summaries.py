from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from pathlib import Path
import json
from typing import List
from fastapi import HTTPException
from config import settings

# Method to summarize texts and tables with error handling
def get_table_summaries(texts: List[str], tables: List[str]):
    try:
        # Initialize the model, prompt, and summarization chain
        prompt_text = """You are an assistant tasked with summarizing tables and text. \
        Give a concise summary of the table or text. Table or text chunk: {element} """
        prompt = ChatPromptTemplate.from_template(prompt_text)

        model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
        summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error initializing GPT model or summarization chain: {str(e)}")

    try:
        # Check if output file exists
        output_file = Path(settings.TABLE_SUMMARIES_PATH)

        # Text Summaries
        text_summaries = texts  # Assume this is already summarized or can be processed similarly

        # Check if the table summaries already exist in the output file
        if output_file.exists():
            with open(output_file, "r") as f:
                table_summaries = json.load(f)
        else:
            # Run summarization for the tables
            table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})

            # Save the summaries to the JSON file
            with open(output_file, "w") as f:
                json.dump(table_summaries, f)

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Error loading or parsing JSON file: {str(e)}")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during table summarization or saving: {str(e)}")

    # Return the summaries
    return text_summaries, table_summaries
    
