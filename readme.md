## Multi-modal RAG

Many documents contain a mixture of content types, including text and images.

Yet, information captured in images is lost in most RAG applications.

With the emergence of multimodal LLMs, like [GPT-4o](https://openai.com/research/gpt-4o-system-card), it is worth considering how to utilize images in RAG:


### What we will be Using / Doing

* Use a multimodal LLM (such as [GPT-4V](https://openai.com/research/gpt-4v-system-card), [LLaVA](https://llava.hliu.cc/), or [FUYU-8b](https://www.adept.ai/blog/fuyu-8b)) to produce text summaries from images
* Embed and retrieve image summaries with a reference to the raw image
* Pass raw images and text chunks to a multimodal LLM for answer synthesis   

This Web APP project highlights `Option 3` from the image shown below.

* We will use [Unstructured](https://unstructured.io/) to parse images, text, and tables from documents (PDFs).
* We will use the [multi-vector retriever](https://python.langchain.com/docs/modules/data_connection/retrievers/multi_vector) with [Chroma](https://www.trychroma.com/) to store raw text, tables and images along with their summaries for retrieval.

![multimodal graph](data\readme_images\multimodal_graph.png)


## SetUp
The PDF partitioning used by Unstructured will use:

- tesseract for Optical Character Recognition (OCR)
- poppler for PDF rendering and processing

Refer to poppler [installation instructions](https://pdf2image.readthedocs.io/en/latest/installation.html) and tesseract [installation instructions](https://tesseract-ocr.github.io/tessdoc/Installation.html) in your system.

**The front_end is only to showcase the results and shows you how to integrate the API's.**


## Developer Guide for environment setup
A minimum of 12 gb memory gpu is required. I used Nvidia RTX 3060 in the development and test process.


### Step 1: Clone the Repository

First, clone the repository to your local machine using the command:

```bash
git clone https://github.com/Abdulkadir19997/Auto-Inpaint-Anything-WebAPP.git
```

**Keep the project archtiecture the same:**
```
├── app
│   ├── ai_services
│   │   ├── create_image_summaries.py
│   │   ├── create_table_summaries.py
│   │   ├── preprocess_vector_db.py
│   │   ├── query_rag_chain.py
│   │   ├── render_preprocess_pdf.py
│   │   ├── __init__.py
│   ├── routes
│   │   ├── inference.py
│   │   ├── preprocess.py
│   │   ├── __init__.py
│   ├── schemas
│   │   ├── schemas.py
│   │   ├── __init__.py
│   ├── __init__.py
├── config.py
├── data
│   ├── chroma_langchain_db
│   │   ├── c4cb8f68-5169-4085-aede-44008a3a4ca7
│   │   ├── chroma.sqlite3
│   ├── content
│   │   ├── image_summaries.json
│   │   ├── table_summaries.json
│   ├── docstore.pkl
│   ├── readme_images
│   │   ├── multimodal_graph.png
├── front_end.py
├── main.py
├── readme.md
├── requirements.txt
├── .env
├── .gitignore
├── __init__.py
```

### Step 2: Create Python Environment

Inside the downloaded 'Automated-Inpaint-Anything' folder, create a Python environment, **I used 3.10.12 version of python**. For example, to create an environment named 'auto_inpainter', use:

```bash
python -m venv auto_inpainter
```

### Step 3: Activate Environment

Activate the environment with:

**For Windows**
```bash
.\auto_inpainter\Scripts\activate
```

**For Linux**
```bash
source auto_inpaint/bin/activate
```

### Step 4: Install Requirements

After confirming that the auto_inpainter environment is active, install all necessary libraries from the 'requirements.txt' file:

```bash
pip install -r requirements.txt
```

### Step 5: Download the checkpoint models from the drive link
Download the model checkpoints:
1. Download the SAM sam_vit_h_4b8939.pth and the big-lama models from the given drive [pretrained_models](https://drive.google.com/drive/folders/1wpY-upCo4GIW4wVPnlMh_ym779lLIG2A?usp=sharing) 
2. Put both of them into `inpaint_anything/pretrained_models` folder (remember to extract the downloaded big-lama.zip file).


### Step 6: Run the Streamlit Application

In the active 'auto_inpainter' environment, run the 'front_end.py' file with:

```bash
streamlit run front_end.py
```

### Step 7: Open a New Terminal Session

Open a new terminal inside the 'Automated-Inpaint-Anything' folder and activate the 'auto_inpainter' environment again:

**For Winows**
```bash
.\auto_inpainter\Scripts\activate
```

**For Linux**
```bash
source auto_inpaint/bin/activate
```

### Step 8: Run FastAPI

In the second terminal with the 'auto_inpainter' environment active, start the FastAPI service with:

```bash
uvicorn main:app --host 127.0.0.1 --port 5004 --reload
```


## Notes
To run locally, operate two different terminals each time: one with the 'auto_inpainter' environment to run 'streamlit run front_end.py', and another to execute 'uvicorn main:app --host 127.0.0.1 --port 5004 --reload'.

## Acknowledgments
Many thanks to these excellent opensource projects
* [LaMA](https://github.com/saic-mdal/lama)
* [GroundingDino](https://huggingface.co/docs/transformers/en/model_doc/grounding-dino)
* [SAM](https://github.com/facebookresearch/segment-anything)
* [Inpaint-Anything](https://github.com/geekyutao/Inpaint-Anything)

## Version
The current version is 1.0. Development is ongoing, and any support or suggestions are welcome. Please reach out to me:
Abdulkadermousabasha7@gmail.com & LinkedIn