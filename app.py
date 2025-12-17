import streamlit as st
from application import (extract_video_id,
                         youtube_transcript,
                         translate_transcript,
                         get_important_topics,
                         generate_notes,
                         create_chunks,
                         creat_vector_store,
                         rag_answer)

with st.sidebar:
    st.title("Yt AI")
    st.markdown("---")
    st.markdown("Transform any youtube video into a key topic, or a chatbot")
    st.markdown("# Input Details")
    yt_url=st.text_input("Past YouTube Video URL",placeholder="https://www.youtube.com/watch?v=0YdpwSYMY6I")
    lan_ipt=st.text_input("Video language code",placeholder="e.g. en, hi, es, fr",value="en")

    task_option=st.radio(
        "Choose what you want to perform :",
        ["Chat With Video","Video Notes For You"]
    )

    btn=st.button("⚡ Start Process")
    st.markdown("---")


st.title("YouTube Content Synthesizer")
st.markdown("Past video link and select task from sidebar")

if btn:
    if yt_url and lan_ipt:
        video_id=extract_video_id(yt_url)
        if video_id:
            with st.spinner("Step 1/3: Extracting Transcript....."):
                full_transcript=youtube_transcript(video_id,lan_ipt)
                if lan_ipt!="en":
                    with st.spinner("Step 1.5/3: Translating Transcript into English....."):
                        full_transcript=translate_transcript(full_transcript)

            if task_option=="Video Notes For You":
                with st.spinner("Step 2/3: Extracting Important Topics For You....."):
                    topic=get_important_topics(full_transcript)
                    st.subheader("Important Topic")
                    st.write(topic)
                with st.spinner("Step 3/3: Writing Notes For You....."):
                    notes=generate_notes(full_transcript)
                    st.subheader("Important Notes")
                    st.write(notes)

                st.success("Summary and Notes generated")

            if task_option=="Chat With Video":
                with st.spinner("Step 2/3: Crating chunks and embeddings...." ):

                    chunks= create_chunks(full_transcript)
                    vectorstore=creat_vector_store(chunks)
                    st.session_state.vector_store=vectorstore
                st.session_state.messages=[]
                st.success("Video is ready to chat!!")

if task_option=="Chat With Video" and "vector_store" in st.session_state:
    st.divider()
    st.subheader("Chat with video")


    for msg in st.session_state.get('messages',[]):
        with st.chat_message(msg['role']):
            st.write(msg['content'])

    prompt=st.chat_input("Ask me anything about the video")
    if prompt:
        st.session_state.messages.append({'role':'user','content':prompt})
        with st.chat_message('user'):
            st.write(prompt)

        with st.chat_message('assistant'):

            response=rag_answer(prompt,st.session_state.vector_store)
            st.write(response)
        st.session_state.messages.append({'role':'assistant','content':response})


































