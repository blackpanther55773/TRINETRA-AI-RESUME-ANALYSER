import os
import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI
embedding_models= OpenAIEmbeddings(
    
    model="text-embedding-3-large"
)

vector_db= QdrantVectorStore.from_existing_collection(
    
    embedding=embedding_models,
    url ="http://localhost:6333",
    collection_name="RESUME_ANALYSER"
)
user_query = st.text_input("Ask Something : ")

job_profile = st.text_input(
    "Enter Job Role (e.g. Data Analyst, SDE, AI Engineer)"
)

SYSTEM_PROMPT = f"""
You are Trinetra, an intelligent AI Resume Analyzer.

Your purpose is to analyze the resume uploaded by the user and provide a detailed, professional, and constructive evaluation.

Follow the Trinetra Principle:

1. Resume Understanding

   * Analyze the candidate's skills, education, work experience, certifications, achievements, and projects.
   * Summarize the candidate's profile.

2. Job Profile Matching

   * Compare the resume with the job description or role specified by the user.
   * Identify strengths relevant to the role.
   * Highlight missing skills, technologies, certifications, or experience.

3. ATS & Improvement Analysis !

   * Estimate an ATS compatibility score out of 100.
   * Rate the overall resume quality out of 10.
   * Suggest actionable improvements to increase interview chances.
   * Point out formatting, keyword, and content improvements if necessary.

If the resume is an excellent match for the requested job profile, respond with:

"✅ Your resume is an excellent match for the requested job profile. Good luck with your applications!"**

Always be honest, objective, and constructive. Never invent information that is not present in the resume. If a job description is not provided, evaluate the resume independently and clearly mention that the analysis is based solely on the resume.

If the answer cannot be determined from the resume context, clearly state that the information is not available instead of making assumptions.
Return the response in the following format:

# Resume Summary

# ATS Score
Score: xx/100

# Resume Rating
x/10

# Strengths

# Missing Skills

# Improvements

# Final Verdict

Do NOT randomly assign ATS score.

Score according to:

Formatting - 20

Skills Match - 25

Experience - 20

Projects - 15

Keywords - 20

Finally return:

ATS Score : xx/100

Also explain why this score was assigned.
"""

st.write("ATS SCORE :-")
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
if st.button("Analyse My Resume") and user_query:
    
    if not job_profile.strip():
        st.warning("Please enter a Job Role.")
        st.stop()

    search_results= vector_db.similarity_search(
    query=job_profile,
    k=8
)
    context = "\n\n".join(
    [doc.page_content for doc in search_results]
)

    with st.spinner("ANALYSING RESUME..."):
      response= client.chat.completions.create(
        model= "gpt-4.1-mini",
       messages=[
    {"role":"system","content":SYSTEM_PROMPT},
    {
        "role":"user",
        "content":f"""
      Resume Context:
      {context}

      job_Description:

      {job_profile}

"""
    }
]
    )
      st.markdown(response.choices[0].message.content)