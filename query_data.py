import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def main():
    # Load environment variables from .env file
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables.")

    # Initialize OpenAIEmbeddings and Chroma
    embedding_function = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Streamlit interface
    st.title("OpenAI Chatbot with Chroma DB")
    query_text = st.text_input("Enter your question:")

    if st.button("Ask"):
        if query_text:
            # Search the DB
            results = db.similarity_search_with_relevance_scores(query_text, k=3)
            if len(results) == 0 or results[0][1] < 0.7:
                st.warning("Unable to find matching results.")
                return

            context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
            prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
            prompt = prompt_template.format(context=context_text, question=query_text)

            # Initialize ChatOpenAI
            model = ChatOpenAI()
            response_text = model.predict(prompt)

            # sources = [doc.metadata.get("source", None) for doc, _score in results]
            st.write(f"Response: {response_text}")
            # st.write(f"Sources: {sources}")
        else:
            st.warning("Please enter a question.")

if __name__ == "__main__":
    main()
