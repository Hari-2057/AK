import streamlit as st
import tempfile
import os
import re
import json
from collections import Counter

# --- Dependency Check & Fail-Safe ---
try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sentence_transformers import SentenceTransformer
    import PyPDF2
    import spacy
    HAS_AI = True
except ImportError:
    HAS_AI = False

# --- UI Setup ---
st.set_page_config(
    page_title="AI Resume Matcher Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

try:
    local_css("style.css")
except:
    pass # Fallback if style.css is not found yet

# --- Constants & DB ---
SKILLS_DB = {
    "Technical": [
        "Python", "Java", "C++", "JavaScript", "TypeScript", "HTML", "CSS", "React", "Angular", "Vue",
        "Node.js", "Django", "Flask", "SQL", "PostgreSQL", "MySQL", "MongoDB", "NoSQL",
        "Docker", "Kubernetes", "AWS", "Azure", "GCP", "CI/CD", "Git", "GitHub",
        "Machine Learning", "Deep Learning", "Data Science", "NLP", "Computer Vision",
        "scikit-learn", "pandas", "numpy", "matplotlib", "seaborn", "TensorFlow", "PyTorch"
    ],
    "Soft Skills": [
        "Communication", "Teamwork", "Leadership", "Agile", "Scrum", "Management",
        "Problem Solving", "Critical Thinking", "Adaptability", "Creativity"
    ],
    "Tools": [
        "Excel", "Power BI", "Tableau", "Jira", "Figma", "Photoshop", "Postman", "Slack"
    ]
}

FLATTENED_SKILLS = [skill for sublist in SKILLS_DB.values() for skill in sublist]

# --- Helper Functions ---

def extract_text_from_pdf(uploaded_file):
    text = ""
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        for page in reader.pages:
            text += page.extract_text() or ""
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return text

def get_clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_skills(text):
    found_skills = set()
    text_lower = text.lower()
    for skill in FLATTENED_SKILLS:
        if re.search(r'\b' + re.escape(skill.lower()) + r'\b', text_lower):
            found_skills.add(skill)
    return found_skills

def calculate_match(resume_text, jd_text):
    if HAS_AI:
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = model.encode([resume_text, jd_text])
            score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return round(score * 100, 2)
        except Exception:
            return 0.0
    else:
        # Fallback to Jaccard for Lite Mode
        r_set = set(resume_text.lower().split())
        j_set = set(jd_text.lower().split())
        if not j_set: return 0.0
        return round((len(r_set & j_set) / len(j_set)) * 100, 2)

def ats_check(text):
    checks = {
        "Contact Info": bool(re.search(r'[\w\.-]+@[\w\.-]+\.\w+|(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text)),
        "Education": bool(re.search(r'Education|Bachelors|Masters|PhD|Degree|University|College', text, re.I)),
        "Experience": bool(re.search(r'Experience|Work History|Employment|Professional Background', text, re.I)),
        "Skills": bool(re.search(r'Skills|Competencies|Expertise|Technologies', text, re.I)),
    }
    score = sum(checks.values()) / len(checks) * 100
    return checks, round(score, 2)

def predict_job_role(skills):
    if not skills: return "Unknown"
    roles = {
        "Data Scientist": ["Python", "Machine Learning", "Data Science", "SQL", "Pandas"],
        "Backend Developer": ["Python", "Java", "Node.js", "Django", "SQL", "Docker"],
        "Frontend Developer": ["React", "JavaScript", "HTML", "CSS", "TypeScript"],
        "Cloud Engineer": ["AWS", "Azure", "Docker", "Kubernetes", "GCP"],
        "AI Engineer": ["Deep Learning", "NLP", "TensorFlow", "PyTorch", "Python"]
    }
    counts = {}
    for role, role_skills in roles.items():
        match_count = len(set(skills) & set(role_skills))
        counts[role] = match_count
    
    best_role = max(counts, key=counts.get)
    if counts[best_role] == 0: return "General Professional"
    return best_role

# --- UI Components ---

def render_hero():
    st.markdown('<div class="hero-text animate-fade-in">Resume Insight Engine</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #94a3b8; font-size: 1.2rem; margin-bottom: 3rem;">AI-Powered ATS Optimization & Resume Matching</p>', unsafe_allow_html=True)

def render_metric_card(label, value, delta=None):
    st.markdown(f"""
    <div class="custom-card">
        <p style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 0.5rem; font-weight: 500;">{label}</p>
        <p style="color: #3b82f6; font-size: 2.2rem; font-weight: 800; margin: 0;">{value}</p>
    </div>
    """, unsafe_allow_html=True)

def render_skill_pills(skills, type="match"):
    class_name = "skill-tag" if type == "match" else "skill-tag skill-tag-missing"
    html = '<div style="margin-top: 10px;">'
    for skill in sorted(list(skills)):
        html += f'<span class="{class_name}">{skill}</span>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

# --- Main App Logic ---

def main():
    render_hero()

    # Sidebar: Input Section
    st.sidebar.markdown("### 📥 Input Data")
    
    use_demo = st.sidebar.toggle("Use Sample Data", value=False)
    
    if use_demo:
        try:
            with open("dummy_jd.txt", "r") as f: demo_jd = f.read()
            with open("sample_resume_content.txt", "r") as f: demo_resume = f.read()
            job_description = st.sidebar.text_area("Job Description", value=demo_jd, height=200)
            resume_input = st.sidebar.text_area("Resume Content", value=demo_resume, height=200)
            file_uploaded = None
        except:
            st.error("Demo files missing. Please use manual input.")
            job_description = st.sidebar.text_area("Job Description", height=200)
            file_uploaded = st.sidebar.file_uploader("Upload Resume (PDF)", type=["pdf"])
            resume_input = ""
    else:
        job_description = st.sidebar.text_area("Target Job Description", height=200, placeholder="Paste the job description here...")
        st.sidebar.markdown("---")
        upload_type = st.sidebar.radio("Resume Input Method", ["Upload PDF", "Paste Text"])
        
        if upload_type == "Upload PDF":
            file_uploaded = st.sidebar.file_uploader("Upload your Resume", type=["pdf"])
            resume_input = ""
        else:
            resume_input = st.sidebar.text_area("Paste Resume Text", height=200)
            file_uploaded = None

    analyze_btn = st.sidebar.button("🔥 Run AI Analysis")

    if analyze_btn:
        if not job_description:
            st.warning("Please provide a Job Description.")
            return

        resume_text = ""
        if file_uploaded:
            with st.spinner("Extracting PDF Text..."):
                resume_text = extract_text_from_pdf(file_uploaded)
        else:
            resume_text = resume_input

        if not resume_text.strip():
            st.error("Please provide your resume content (PDF or Text).")
            return

        # Start Processing
        with st.spinner("💡 AI Engine is analyzing your profile..."):
            resume_clean = get_clean_text(resume_text)
            jd_clean = get_clean_text(job_description)
            
            # 1. Match Score
            match_score = calculate_match(resume_clean, jd_clean)
            
            # 2. ATS Score
            ats_details, ats_score = ats_check(resume_text)
            
            # 3. Skills Extraction
            resume_skills = extract_skills(resume_text)
            jd_skills = extract_skills(job_description)
            missing_skills = jd_skills - resume_skills
            matching_skills = resume_skills & jd_skills
            
            # 4. Role Prediction
            predicted_role = predict_job_role(resume_skills)

            # --- Layout: Main Results ---
            col1, col2, col3 = st.columns(3)
            with col1:
                render_metric_card("Overall Match", f"{match_score}%")
            with col2:
                render_metric_card("ATS Score", f"{ats_score}%")
            with col3:
                render_metric_card("Profile Match", predicted_role)

            st.markdown("---")

            # --- Layout: Details ---
            tab1, tab2, tab3 = st.tabs(["🎯 Skill Analysis", "📋 ATS Checklist", "🔍 Keyword Insights"])

            with tab1:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("### ✅ Matching Skills")
                    if matching_skills:
                        render_skill_pills(matching_skills, "match")
                    else:
                        st.info("No matching skills detected from our database.")
                
                with c2:
                    st.markdown("### ❌ Skill Gaps")
                    if missing_skills:
                        render_skill_pills(missing_skills, "missing")
                        st.warning("To improve your score, consider adding these skills if you have them.")
                    else:
                        st.success("Perfect Skills Match! No gaps found.")

            with tab2:
                st.markdown("### ATS Format Scan")
                for item, status in ats_details.items():
                    icon = "✅" if status else "❌"
                    color = "#10b981" if status else "#ef4444"
                    st.markdown(f'<div style="font-size: 1.1rem; margin-bottom: 8px;">{icon} <span style="font-weight:600">{item}:</span> <span style="color: {color}">{"Perfect" if status else "Missing / Not Found"}</span></div>', unsafe_allow_html=True)
                
                if ats_score < 100:
                    st.info("💡 Pro Tip: Ensure your resume has clear headers for Education and Experience to pass ATS filters.")

            with tab3:
                st.markdown("### Top Keywords in Resume")
                words = re.findall(r'\b\w{4,}\b', resume_text.lower())
                stop_words = {'with', 'from', 'this', 'that', 'your', 'have', 'experience', 'worked', 'project'}
                filtered_words = [w for w in words if w not in stop_words]
                word_counts = Counter(filtered_words).most_common(10)
                
                # Simple visual representation
                for word, count in word_counts:
                    st.markdown(f"""
                    <div style="display: flex; align-items: center; margin-bottom: 10px;">
                        <div style="width: 100px; font-weight: 500;">{word.capitalize()}</div>
                        <div style="flex-grow: 1; background: #1e293b; border-radius: 10px; height: 10px;">
                            <div style="width: {min(100, count*10)}%; background: var(--primary-gradient); height: 100%; border-radius: 10px;"></div>
                        </div>
                        <div style="width: 40px; text-align: right; margin-left: 10px; font-size: 0.8rem; color: #94a3b8;">{count}</div>
                    </div>
                    """, unsafe_allow_html=True)

    else:
        # Initial Landing State
        st.markdown("""
        <div class="custom-card" style="text-align: center; border: 1px dashed rgba(255,255,255,0.2); background: transparent;">
            <p style="font-size: 1.5rem; margin-top: 2rem;">👋 Ready to land your dream job?</p>
            <p style="color: #94a3b8;">Upload your resume and the job description in the sidebar to get started.</p>
            <div style="height: 100px;"></div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
