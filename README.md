Output:filtered_articles.json(hybrid(keyword+semantic) search),extracted_insights.json are present in the output folder

Setup: Install all necessary Python packages:
1. Install Dependencies after creating and activating a virtual environments.
`**_pip install -r requirements.txt_**`
-Replace the Groq API Key in the .env file.
2. Filter Articles
Filter Articles: Run the first script (filter_articles.py) to identify relevant articles based on keywords (e.g., "cancer", "immunology") and semantic similarity.
This reads Markdown files from ../data/papers/ and creates ../outputs/filtered_articles.json containing a list of relevant PubMed IDs.
**_`python filter_articles.py`_**
3. Extract Insights
Uses Groq + LangGraph to extract structured biomedical info.
Extract Insights: Run the second script (extract_insights.py). 
This script reads the list of IDs from filtered_articles.json, fetches the corresponding article text, uses the Groq LLM via LangGraph to extract structured data (diseases, genes, etc.) and generate summaries, and finally saves the combined results into ../outputs/extracted_insights2.json.

**_`python extract_insights.py`_**

# **Approach:**
### **filter_articles.py**

This script filter_articles.py processes a directory of scientific articles (in Markdown format, named by their PubMed ID) to identify and filter articles related to specific topics (currently "cancer" and "immunology"). 
It uses a combination of keyword matching and semantic similarity search (using Sentence Transformers and FAISS) to achieve this. The final output is a JSON file containing a list of PubMed IDs for the filtered articles.

## Features

* Reads Markdown files from a specified input directory.
* Assumes filenames correspond to PubMed IDs (e.g., `12345678.md`).
* Extracts plain text content from Markdown articles.
* Performs keyword matching for terms like "cancer" and "immunology" (case-insensitive, allows variations like "cancers", "immunological").
* Generates sentence embeddings for all article texts using a pre-trained Sentence Transformer model (`all-MiniLM-L6-v2`).
* Builds a FAISS index for efficient semantic similarity search.
* Performs a semantic search using a general query related to the topics ("This is a text related to cancer or immunology").
* Filters semantic search results based on a cosine similarity threshold (currently > 0.2).
* Combines the results from keyword matching and semantic search (union of both sets).
* Saves the final list of filtered PubMed IDs to a JSON file in a specified output directory.

## Prerequisites

* Input articles in Markdown format (`.md`) located in a specific directory (default: `../data/papers/`).
* An output directory must exist (default: `../outputs/`).

**### Biomedical Article Insight Extractor using LangGraph**

This script processes a pre-filtered list of scientific articles (identified by PubMed IDs) to extract key biomedical insights using Large Language Models (LLMs) accessed via the Groq API. 
It employs LangGraph to orchestrate a workflow that first extracts structured entities (Diseases, Genes/Proteins, Pathways, Experimental Methods) and
then generates a concise summary focusing on the main scientific findings for each article.

The script relies on the output of a prior filtering step ( performed by `filter_articles.py`) which provides a list of relevant PubMed IDs.

## Features

* Loads article text for PubMed IDs specified in a JSON input file (`filtered_articles.json`).
* Uses the Groq API for fast LLM inference (configured for `llama-3.1-8b-instant`).
* Leverages LangGraph to define and execute a multi-step analysis workflow:
    * **Structured Extraction Node:** Uses a Groq LLM prompted for structured JSON output (conforming to a Pydantic schema) to identify specific biomedical entities.
    * **Summarization Node:** Uses a Groq LLM prompted to generate a brief (1-2 sentence) summary highlighting the core scientific insight.
* Orchestrates the nodes sequentially (Extraction -> Summarization).
* Utilizes Pydantic for defining data schemas (`Features`, `CombinedOutput`) ensuring structured and validated outputs from the LLM.
* Loads the required Groq API key securely from a `.env` file.
* Transforms the combined output (structured data + summary) into a flattened dictionary format for the final JSON.
* Saves the extracted insights for each article, keyed by PubMed ID, into a final JSON file (`extracted_insights2.json`).

# **Evaluation of LLM Results:**

Overall, the LLM performed reasonably well in extracting the key entities,
especially Diseases, Genes/Proteins involved, 
and Experimental Methods used, often aligning closely with the Key_Findings. 
1.The LLM sometimes identified a technology or biomarker as the disease (e.g., "CRISPR/Cas9 gene-editing technologies" in 40242591
2. The LLM sometimes misses out on detecting pathways in some cases.