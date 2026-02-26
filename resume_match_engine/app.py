
import streamlit as st
import tempfile
import os
import re

# --- Dependency Check & Fail-Safe ---
try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sentence_transformers import SentenceTransformer
    import PyPDF2
    import spacy
    HAS_AI = True
except ImportError:
    HAS_AI = False
    st.warning("⚠️ Running in Lite Mode: AI libraries are still installing. Using basic analysis for now.")

# Load spaCy model (small one for efficiency)
nlp = None
if HAS_AI:
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        nlp = None
    except NameError:
        nlp = None

# Predefined skills list for simple extraction
SKILLS_DB = {
    "Python", "Java", "C++", "JavaScript", "TypeScript", "HTML", "CSS", "React", "Angular", "Vue",
    "Node.js", "Django", "Flask", "SQL", "PostgreSQL", "MySQL", "MongoDB", "NoSQL",
    "Docker", "Kubernetes", "AWS", "Azure", "GCP", "CI/CD", "Git", "GitHub",
    "Machine Learning", "Deep Learning", "Data Science", "NLP", "Computer Vision",
    "scikit-learn", "pandas", "numpy", "matplotlib", "seaborn", "TensorFlow", "PyTorch",
    "Communication", "Teamwork", "Leadership", "Agile", "Scrum",
    "Excel", "Power BI", "Tableau"
}

def extract_text_from_pdf(uploaded_file):
    text = ""
    if HAS_AI:
        try:
            reader = PyPDF2.PdfReader(uploaded_file)
            for page in reader.pages:
                text += page.extract_text() or ""
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
    else:
        st.error("PDF reading unavailable in Lite Mode (PyPDF2 missing). Please paste text.")
    return text

def extract_skills(text):
    found_skills = set()
    text_lower = text.lower()
    for skill in SKILLS_DB:
        if re.search(r'\b' + re.escape(skill.lower()) + r'\b', text_lower):
            found_skills.add(skill)
    return found_skills

def calculate_similarity(resume_text, job_desc_text):
    if HAS_AI:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode([resume_text, job_desc_text])
        score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return round(score * 100, 2)
    else:
        # Simple Jaccard Similarity Mock for Lite Mode
        resume_words = set(resume_text.lower().split())
        job_words = set(job_desc_text.lower().split())
        intersection = resume_words.intersection(job_words)
        union = resume_words.union(job_words)
        if not union:
            return 0.0
        return round((len(intersection) / len(union)) * 100 * 2, 2) # boost score for demo

def extract_years_experience(text):
    match = re.search(r'(\d+)\+?\s*years?', text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 0

st.set_page_config(page_title="AI Resume Matcher", layout="wide")

st.title("🚀 AI Resume Matcher & Skill Gap Analyzer")
st.markdown("Build your resume to match the job description perfectly.")

# Sidebar
st.sidebar.header("Upload Data")

use_demo_data = st.sidebar.checkbox("Use Demo Data (for Presentation)")

if use_demo_data:
    try:
        with open("dummy_jd.txt", "r") as f:
            demo_jd = f.read()
        with open("sample_resume_content.txt", "r") as f:
            demo_resume = f.read()
            
        resume_source = "Paste Text"
        resume_paste_text = st.sidebar.text_area("Resume Text", value=demo_resume, height=300)
        job_description = st.sidebar.text_area("Job Description", value=demo_jd, height=300)
        uploaded_resume = None
    except FileNotFoundError:
        st.error("Demo files not found. Please upload manually.")
        resume_source = st.sidebar.radio("How to provide resume?", ("Upload PDF", "Paste Text"))
        if resume_source == "Upload PDF":
            uploaded_resume = st.sidebar.file_uploader("Upload Resume (PDF)", type=["pdf"])
        else:
            uploaded_resume = None
        resume_paste_text = ""
        if resume_source == "Paste Text":
            resume_paste_text = st.sidebar.text_area("Paste Resume Text", height=300)
        job_description = st.sidebar.text_area("Paste Job Description", height=300)
else:
    resume_source = st.sidebar.radio("How to provide resume?", ("Upload PDF", "Paste Text"))
    if resume_source == "Upload PDF":
        uploaded_resume = st.sidebar.file_uploader("Upload Resume (PDF)", type=["pdf"])
    else:
        uploaded_resume = None

    resume_paste_text = ""
    if resume_source == "Paste Text":
        resume_paste_text = st.sidebar.text_area("Paste Resume Text", height=300)
        
    job_description = st.sidebar.text_area("Paste Job Description", height=300)

if st.sidebar.button("Analyze Resume"):
    # Check if we have valid input
    has_resume = (uploaded_resume is not None) or (resume_source == "Paste Text" and resume_paste_text.strip())
    
    if has_resume and job_description:
        with st.spinner("Analyzing resume..."):
            if resume_source == "Upload PDF" and uploaded_resume:
                resume_text = extract_text_from_pdf(uploaded_resume)
            else:
                resume_text = resume_paste_text
            
            if not resume_text.strip():
                st.error("Could not extract text from the resume.")
            else:
                # 1. Match Score
                match_score = calculate_similarity(resume_text, job_description)
                
                # 2. Skill Extraction
                resume_skills = extract_skills(resume_text)
                job_skills = extract_skills(job_description)
                
                missing_skills = job_skills - resume_skills
                strong_areas = resume_skills.intersection(job_skills)
                
                # 3. Experience Gap (Simple Check)
                resume_exp = extract_years_experience(resume_text)
                job_exp = extract_years_experience(job_description)
                exp_gap = max(0, job_exp - resume_exp)

                # --- Display Results ---
                
                col1, col2, col3 = st.columns(3)
                
                # Match Score Card
                with col1:
                    st.metric(label="Match Score", value=f"{match_score}%")
                    if match_score >= 80:
                        st.success("Great Match!")
                    elif match_score >= 60:
                        st.warning("Good, but improvements needed.")
                    else:
                        st.error("Low match score.")

                # Experience Gap
                with col3:
                    st.metric(label="Experience Gap", value=f"{exp_gap} Years" if exp_gap > 0 else "Analysis Inconclusive / No Gap")
                
                st.markdown("---")
                
                # Skills Analysis
                c1, c2 = st.columns(2)
                
                with c1:
                    st.subheader("✅ Strong Areas (Matching Skills)")
                    if strong_areas:
                        for skill in strong_areas:
                            st.markdown(f"- **{skill}**")
                    else:
                        st.info("No direct skill matches found based on predefined list.")
                        
                with c2:
                    st.subheader("❌ Missing Skills (Gap Analysis)")
                    if missing_skills:
                        for skill in missing_skills:
                            st.markdown(f"- <span style='color:red'>{skill}</span>", unsafe_allow_html=True)
                    else:
                        st.success("No missing skills found!")

                # Expandable Details
                with st.expander("See Extracted Resume Text"):
                    st.text(resume_text)

    else:
        st.warning("Please upload a resume and paste a job description.")

st.markdown("---")
st.caption("AI Resume Matcher - Prototype v1.0")
