import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
from datasets import Dataset, Features, Sequence, Value
from ragas import evaluate
from ragas.metrics import faithfulness, answer_correctness
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import random
import string

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
    st.title("Document Reader ChatBot")
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

            data_samples = {
                'question': [
                    query_text
                ],
                'answer': [
                    response_text
                ],
                'contexts': [
                    [context_text]
                ],
                'ground_truth': [
                    'Alice Adventures in Wonderland Author is Lewis Carroll'
                ]
            }
            features = Features({
            'question': Value('string'),
            'answer': Value('string'),
            'contexts': Sequence(Value('string')),  # Ensure contexts is a sequence of strings
            'ground_truth': Value('string')
             })
            dataset = Dataset.from_dict(data_samples, features=features)

            # Print the dataset to verify the structure
            print(dataset)

            score = evaluate(dataset, metrics=[faithfulness, answer_correctness])
            print(score)
            df = score.to_pandas()
            df.to_csv('score.csv', index=False)
            # sources = [doc.metadata.get("source", None) for doc, _score in results]
            st.write(f"Response: {response_text}")
            # st.write(f"Sources: {sources}")


            def normalize_text(text):
                text = text.strip().lower()
                # Remove punctuation
                text = re.sub(r'[^\w\s]', '', text)
                return text
            
            relevant_contexts = [
                "the author of Alice's Adventures in Wonderland Author is Lewis Carroll Release date: June 27, 2008..."
            ]
            def extract_keywords(context):
                # Extract keywords or key phrases from the context
                keywords = [
                    "author lewis carroll",
                    "release date june 27 2008"
                ]
                return keywords

            def calculate_precision(retrieved_contexts, relevant_contexts):
                relevant_retrieved = []
                for context in retrieved_contexts:
                    for rel_context in relevant_contexts:
                        keywords = extract_keywords(rel_context)
                        if any(keyword in normalize_text(context) for keyword in keywords):
                            relevant_retrieved.append(context)
                            break
                print("Relevant Retrieved Contexts:", relevant_retrieved)
                precision = len(relevant_retrieved) / len(retrieved_contexts) if retrieved_contexts else 0
                return precision
            
            def calculate_recall(retrieved_contexts, relevant_contexts):
                relevant_retrieved = []
                normalized_relevant_contexts = [normalize_text(context) for context in relevant_contexts]
                retrieved_set = set(retrieved_contexts)
                for rel_context in normalized_relevant_contexts:
                    keywords = extract_keywords(rel_context)
                    if any(keyword in retrieved_set for keyword in keywords):
                        relevant_retrieved.append(rel_context)
                print("Relevant Retrieved Contexts:", relevant_retrieved)
                recall = len(relevant_retrieved) / len(relevant_contexts) if relevant_contexts else 0
                return recall
            
            def calculate_context_relevance(retrieved_contexts, query):
                # Vectorize the texts using TF-IDF
                vectorizer = TfidfVectorizer()
                all_texts = retrieved_contexts + [query]
                tfidf_matrix = vectorizer.fit_transform(all_texts)
                
                # Compute cosine similarity between query and retrieved contexts
                query_vector = tfidf_matrix[-1]
                context_vectors = tfidf_matrix[:-1]
                similarities = cosine_similarity(query_vector, context_vectors).flatten()
                
                # Calculate relevance as the average similarity score
                relevance = np.mean(similarities)
                return relevance
            
            # Function to add noise to the context
            def add_noise_to_context(context, noise_level=0.1):
                context_chars = list(context)
                num_chars_to_modify = int(noise_level * len(context_chars))
                for _ in range(num_chars_to_modify):
                    pos = random.randint(0, len(context_chars) - 1)
                    context_chars[pos] = random.choice(string.ascii_letters + string.digits + string.punctuation)
                return ''.join(context_chars)
            
            def calculate_noise_robustness(retrieved_contexts_noisy, relevant_contexts):
                relevant_retrieved = [
                    context for context in retrieved_contexts_noisy 
                    if any(rel_context in normalize_text(context) for rel_context in normalized_relevant_contexts)
                ]
                print("Relevant Retrieved Contexts with Noise:", relevant_retrieved)
                precision_with_noise = len(relevant_retrieved) / len(retrieved_contexts_noisy) if retrieved_contexts_noisy else 0
                return precision_with_noise
            
            def evaluate_counterfactual_robustness(generated_answers, expected_responses):
                correct_responses = sum(1 for gen_ans, exp_resp in zip(generated_answers, expected_responses) if gen_ans == exp_resp)
                robustness_score = correct_responses / len(expected_responses)
                return robustness_score
            
            expected_responses = [
                "lewis carroll"
                 ]

            normalized_relevant_contexts = [normalize_text(context) for context in relevant_contexts]
            retrieved_contexts = [context_text]
            retrieved_contexts = context_text.split('\n')
            retrieved_contexts = [normalize_text(context) for context in retrieved_contexts if context.strip()]
            print("Retrieved Contexts:", retrieved_contexts)
            precision = calculate_precision(retrieved_contexts, relevant_contexts)
            print("Precision:", precision)
            recall = calculate_recall(retrieved_contexts, relevant_contexts)
            print("Recall:", recall)
            context_relevance = calculate_context_relevance(retrieved_contexts, query_text)
            print("Context Relevance:", context_relevance) 
            retrieved_contexts_noisy = [add_noise_to_context(context, noise_level=0.1) for context in retrieved_contexts]
            print("Retrieved Contexts with Noise:", retrieved_contexts_noisy)
            noise_robustness = calculate_noise_robustness(retrieved_contexts_noisy, relevant_contexts)
            print("Noise Robustness (Precision with Noise):", noise_robustness)
            robustness_score = evaluate_counterfactual_robustness([response_text], expected_responses)
            print("Counterfactual Robustness Score:", robustness_score)    
    
        else:
            st.warning("Please enter a question.")

if __name__ == "__main__":
    main()
