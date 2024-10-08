import os
import sys
import time
import subprocess
from dotenv import load_dotenv

import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationSummaryBufferMemory
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Package installation function
def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# List of required packages
REQUIRED_PACKAGES = [
    "streamlit", "langchain", "langchain_community", "langchain-huggingface",
    "faiss-cpu", "openai", "tiktoken"
]

# Prompt template
PROMPT_TEMPLATE = """
You are an AI-powered course recommendation expert with extensive knowledge of educational programs across various disciplines. Your primary goal is to provide personalized, high-quality course suggestions tailored to each user's unique interests, goals, and background.
Do not retrieve course recommendations if the user hasn't specifically asked for them or is simply greeting the chatbot. 
In such general cases, focus on engaging the user by asking about their learning interests or what they are looking to explore.

Conversation History:
{chat_history}

Current User Query:
{question}

Relevant Courses from Database:
{context}

Instructions for Crafting Your Response:
1. Engagement and Tone:
   - Begin with a warm, friendly greeting if this is a new interaction.
   - Maintain a professional yet approachable tone throughout the conversation.
   - If the user initiates casual chat, engage briefly before steering the conversation towards educational interests.
2. Analysis and Recommendation:
   - Carefully analyze the user's query and conversation history to understand their educational needs, interests, and any constraints.
   - Select the most relevant courses from the provided context, prioritizing those with learning outcomes and syllabus content that closely match the user's requirements.
3. Detailed Course Recommendations:
   For each recommended course, provide:
   - Course title and offering institution
   - A concise overview of the course content
   - Specific skills and knowledge to be gained (from "What You Will Learn")
   - Key topics covered in the syllabus
   - Course level, duration, and language of instruction
   - Course ratings and reviews (if available)
   - Direct URL to the course page (If available)
4. Personalized Explanation:
   - Clearly articulate how each recommended course aligns with the user's expressed interests and goals.
   - Highlight specific aspects of the course that address the user's needs or previous queries.

Remember to prioritize accuracy, relevance, and user-centricity in your recommendations. Your goal is to empower the user to make informed decisions about their educational path.

Recommendation:
"""

def setup_qa_chain():
    
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Load the FAISS index
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # Set up the language model and memory
    llm = ChatOpenAI(temperature=0.7, model_name="gpt-4-turbo")
    memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=1000, memory_key="chat_history", return_messages=True)
    
    # Create the prompt
    prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["chat_history", "question", "context"])
    
    # Create and return the conversational retrieval chain
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )

def get_course_recommendations(qa_chain, user_query: str):
    result = qa_chain({"question": user_query})
    return result["answer"]

def main():
    
    load_dotenv()
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    print(OPENAI_API_KEY)

    # Check and install required packages
    for package in REQUIRED_PACKAGES:
        try:
            __import__(package)
        except ImportError:
            print(f"{package} not found. Installing...")
            install_package(package)
    print("All required packages are installed.")
    
    
    # Streamlit app setup
    st.set_page_config(page_title="Course Recommendation Chatbot", page_icon=":book:")
    st.title("HONEY BEE: Course Recommendation Chatbot 🐝")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        welcome_message = (
            "Hello! I'm HONEY BEE, your friendly Course Recommendation Chatbot! 🐝 "
            "I'm here to help you find the best courses based on your interests and goals. "
            "Feel free to ask me anything about learning or courses!"
        )
        st.session_state.messages.append({"role": "assistant", "content": welcome_message})

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Set up QA chain
    qa_chain = setup_qa_chain()

    # Handle user input
    if prompt := st.chat_input("What are you looking to learn?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            response_text = get_course_recommendations(qa_chain, prompt)
            placeholder = st.empty()
            accumulated_response = ""
            for char in response_text:
                accumulated_response += char
                placeholder.markdown(accumulated_response, unsafe_allow_html=True)
                time.sleep(0.01)

        st.session_state.messages.append({"role": "assistant", "content": response_text})

if __name__ == "__main__":
    main()