import tempfile
import os
import streamlit as st
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import re


st.set_page_config(
    page_title="Trinetra AI",
    page_icon=r"C:\Users\harsh\Downloads\ChatGPT Image Jul 4, 2026, 10_09_32 AM.png",
    layout="centered"
)
page_icon="🧿"

st.markdown(
    """
    <h1 style='text-align:center;'>
        🧿 Trinetra AI Resume Analyzer
    </h1>
    <p style='text-align:center; color:gray;'>
        AI Powered Resume Analysis • ATS Score • Job Matching
    </p>
    """,
    unsafe_allow_html=True
)


uploaded_file = st.file_uploader("Upload Your Resume", type="pdf")

job_profile = st.text_input(
    "Enter Job Role (e.g. Data Analyst, SDE, AI Engineer)"
)

embedding_models = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

SYSTEM_PROMPT = """
You are Trinetra, an intelligent AI Resume Analyzer.

Your purpose is to analyze the resume uploaded by the user and provide a detailed, professional, and constructive evaluation.

Follow the Trinetra Principle:

1. Resume Understanding
- Analyze skills, education, work experience, certifications, achievements and projects.
- Summarize the profile.

2. Job Profile Matching
- Compare with requested job role.
- Mention strengths.
- Mention missing skills.

3. ATS Analysis
- Estimate ATS Score /100
- Resume Rating /10
- Improvements
- Final Verdict

Use this format:

# Resume Summary

# ATS Score

# Resume Rating

# Strengths

# Missing Skills

# Improvements

# Final Verdict

Scoring:

Formatting - 20

Skills Match - 25

Experience - 20

Projects - 15

Keywords - 20

Explain why the ATS score was assigned.
"""

st.write("### ATS SCORE")


if st.button("Analyse My Resume..."):

    if uploaded_file is None:
        st.warning("Please upload your resume.")
        st.stop()

    if not job_profile.strip():
        st.warning("Please enter Job Role.")
        st.stop()

    with st.spinner("Uploading & Processing Resume..."):

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_path = temp_file.name

        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=450
        )

        chunks = splitter.split_documents(docs)

        # ✅ FAISS ONLY (NO QDRANT)
        vector_db = FAISS.from_documents(
            chunks,
            embedding_models
        )

    st.success("✅ Resume Uploaded Successfully!")

    search_results = vector_db.similarity_search(
        query=job_profile,
        k=8
    )

    context = "\n\n".join(
        [doc.page_content for doc in search_results]
    )

    # ✅ FIXED API KEY HANDLING
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        st.error("OPENAI API KEY NOT FOUND")
        st.stop()

    client = OpenAI(api_key=api_key)

    with st.spinner("🧠 Analysing Resume..."):

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": f"""
Resume Context:

{context}

Job Description:

{job_profile}
"""
                }
            ]
        )

    st.success("✅ Resume Analysis Completed!")

    response_text = response.choices[0].message.content
    st.markdown(response_text)

    match = re.search(
        r'ATS Score\s*[:\-]?\s*(\d+)',
        response_text,
        re.IGNORECASE
    )

    if match:
        ats_score = int(match.group(1))

        st.subheader("📊 ATS Score")
        st.progress(ats_score / 100)

        if ats_score >= 80:
            st.success(f"Excellent ATS Score: {ats_score}/100")
        elif ats_score >= 60:
            st.warning(f"Good ATS Score: {ats_score}/100")
        else:
            st.error(f"Needs Improvement: {ats_score}/100")


st.divider()

st.markdown(
    """
    <div style="text-align:center;padding:15px;">
        <h4>🧠 Trinetra AI Resume Analyzer</h4>
        <p style="color:gray;">
            Developed by <b>Harsh Kumar</b><br>
            Powered by OpenAI • LangChain • FAISS • Streamlit
        </p>
    </div>
    """,
    unsafe_allow_html=True
)