import re
import streamlit as st
from dotenv import load_dotenv
import time
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
load_dotenv()
api_key=os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Google API not found.")
def extract_video_id(url):
    """
    Extracts the YouTube video ID from any valid YouTube URL.
    """
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    if match:
        return match.group(1)
    st.error("Invalid YouTube URL. Please enter a valid video link.")
    return None

def youtube_transcript(video,language):

    yt_transcript=YouTubeTranscriptApi()
    try:
        transcript=yt_transcript.fetch(video,languages=[language])
        full_transcript=" ".join(i.text for i in transcript)
        time.sleep(10)
        return full_transcript
    except Exception as e:
        st.error(f"Error occure for fetching Youtube video {e}")

# abc=youtube_transcript("0YdpwSYMY6I","en")
# print(abc)
llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0.2
    )
def translate_transcript(transcript):


    try:
        prompt=ChatPromptTemplate.from_template("""
            You are an expert translator with deep cultural and linguistic knowledge.
            I will provide you with a transcript. Your task is to translate it into English with absolute accuracy, preserving:
            - Full meaning and context (no omissions, no additions).
            - Tone and style (formal/informal, emotional/neutral as in original).
            - Nuances, idioms, and cultural expressions (adapt appropriately while keeping intent).
            - Speaker’s voice (same perspective, no rewriting into third-person).
            Do not summarize or simplify. The translation should read naturally in the target language but stay as close as possible to the original intent.
    
            Transcript:
            {transcript}
            """)

        chain=prompt|llm
        response= chain.invoke({"transcript":transcript})

        return response.content
    except Exception as e:
        st.error(f"Error translate transcript {e}")

def get_important_topics(transcript):
    try:
        prompt = ChatPromptTemplate.from_template("""
               You are an assistant that extracts the 5 most important topics discussed in a video transcript or summary.

               Rules:
               - Summarize into exactly 5 major points.
               - Each point should represent a key topic or concept, not small details.
               - Keep wording concise and focused on the technical content.
               - Do not phrase them as questions or opinions.
               - Output should be a numbered list.
               - show only points that are discussed in the transcript.
               Here is the transcript:
               {transcript}
               """)

        # Runnable chain
        chain = prompt | llm

        # Run chain
        response = chain.invoke({"transcript": transcript})

        return response.content

    except Exception as e:
        st.error(f"Error finding important topics {e}")



# FUNCTION TO GET NOTES FROM THE VIDEO
def generate_notes(transcript):
    try:
        prompt = ChatPromptTemplate.from_template("""
                You are an AI note-taker. Your task is to read the following YouTube video transcript 
                and produce well-structured, concise notes.

                ⚡ Requirements:
                - Present the output as **bulleted points**, grouped into clear sections.
                - Highlight key takeaways, important facts, and examples.
                - Use **short, clear sentences** (no long paragraphs).
                - If the transcript includes multiple themes, organize them under **subheadings**.
                - Do not add information that is not present in the transcript.

                Here is the transcript:
                {transcript}
                """)

        # Runnable chain
        chain = prompt | llm

        # Run chain
        response = chain.invoke({"transcript": transcript})

        return response.content

    except Exception as e:
        st.error(f"Error generate notes {e}")


def create_chunks(transcript):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    doc=text_splitter.create_documents([transcript])
    return doc

def creat_vector_store(docs):
    # embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001",transport="grpc")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store=Chroma.from_documents(docs,embeddings)
    return vector_store


def rag_answer(question,vectorstore):
    results=vectorstore.similarity_search(question,k=5)
    context_text="\n".join([i.page_content for i in results])
    prompt = ChatPromptTemplate.from_template("""
                  You are a kind, polite, and precise assistant.
                  - Begin with a warm and respectful greeting (avoid repeating greetings every turn).
                  - Understand the user’s intent even with typos or grammatical mistakes.
                  - Answer ONLY using the retrieved context.
                  - If answer not in context, say:
                    "I couldn’t find that information in the database. Could you please rephrase or ask something else?"
                  - Keep answers clear, concise, and friendly.

                  Context:
                  {context}

                  User Question:
                  {question}

                  Answer:
                  """)

    # chain
    chain = prompt | llm
    response = chain.invoke({"context": context_text, "question": question})

    return response.content

















