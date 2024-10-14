import io
import os
import base64
import json
from pathlib import Path
import shutil
from langchain_openai import ChatOpenAI
from langchain.schema.messages import HumanMessage
from typing import List, Dict
from fastapi import HTTPException
from config import settings

# Helper method to encode image to base64
def encode_image(image_path: str) -> str:
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error encoding image {image_path}: {str(e)}")

# Method to summarize an image using GPT
def image_summarize(img_base64: str, prompt: str) -> str:
    try:
        chat = ChatOpenAI(model="gpt-4o", max_tokens=1024)

        msg = chat.invoke(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_base64}"
                            },
                        },
                    ]
                )
            ]
        )
        return msg.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error summarizing image: {str(e)}")

# Method to summarize all images in a folder and save results to a JSON file
def get_image_summaries(image_folder: str = "figures", prompt: str = "Describe the image in detail. Be specific about graphs, such as bar plots.") -> List[Dict[str, str]]:
    try:
        # Initialize storage for image summaries
        image_summaries = []
        # Store base64 encoded images
        img_base64_list = []
        output_file = Path(settings.IMAGE_SUMMARIES_PATH)

        # Load existing summaries if the file exists
        if output_file.exists():
            with open(output_file, "r") as f:
                image_summaries = json.load(f)
        else:
            image_summaries = []

        # Process each image in the folder
        for img_file in sorted(os.listdir(image_folder)):
            if img_file.endswith('.jpg'):
                img_path = os.path.join(image_folder, img_file)

                # Encode the image to base64 (always append base64 to the list)
                base64_image = encode_image(img_path)
                img_base64_list.append(base64_image)

                # Check if the image is already summarized
                if any(summary['image'] == img_file for summary in image_summaries):
                    continue  # Skip summarization if already summarized
                else:
                    # Summarize the image
                    summary = image_summarize(base64_image, prompt)
                    # Store the summary with a reference to the image
                    image_summaries.append({"image": img_file, "summary": summary})

        # Save the updated summaries to a JSON file
        with open(output_file, "w") as f:
            json.dump(image_summaries, f)

        # Check if the folder exists
        if os.path.exists(image_folder):
            # Remove the entire folder and its contents
            shutil.rmtree(image_folder)
        
        return image_summaries, img_base64_list

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing images: {str(e)}")
