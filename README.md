# Document Reader Chatbot with Chroma DB

## Video Walkthrough

## Contextual Information Retrieval

This Streamlit application integrates an OpenAI-powered chatbot with a Chroma database for contextual information retrieval. Users can ask questions based on pre-indexed documents and receive relevant responses.

Efficient Information Access: Users can quickly retrieve relevant information by asking natural language questions. This saves time compared to manually searching through documents or databases.

Accurate Responses: The chatbot leverages advanced natural language processing (NLP) capabilities from OpenAI to understand and respond to user queries accurately, based on the indexed documents in the Chroma database.

## Use Case Examples
Research Assistance: Researchers can efficiently gather information on specific topics without manually sifting through extensive documents.

Customer Support: Users seeking information or troubleshooting help can receive immediate answers, enhancing customer satisfaction.

Educational Applications: Students and educators can use the chatbot to find detailed explanations or references quickly.

### File Information

## create_database.py

The `create_database.py` utilizes LangChain and OpenAI to load Markdown documents from a specified directory, split them into smaller chunks, and persist them into a Chroma database for efficient contextual information retrieval using an OpenAI-powered chatbot.

## query_data.py

This `query_data.py` integrates an OpenAI-powered chatbot with a Chroma database via Streamlit. Users input questions to retrieve relevant information from pre-indexed document contexts, facilitated by natural language processing.

## Installation

1. Run this command to install dependenies in the `requirements.txt` file. 

```python
pip install -r requirements.txt
```

2. Create database

Create the Chroma DB.

```python
python create_database.py
```

3. Run the streamlit application

```python
streamlit run query_data.py
```



