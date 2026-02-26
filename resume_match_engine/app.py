import streamlit as st
import re
import json
from collections import Counter
# --- Dependency Check & Fail-Safe ---
try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sentence_transformers import SentenceTransformer
    import PyPDF2
    HAS_AI = True
except Exception:
    HAS_AI = False
    # Define a dummy PyPDF2 if missing to avoid NameError later
    import types
    PyPDF2 = types.ModuleType("PyPDF2")

# --- UI Setup ---
st.set_page_config(
    page_title="Resume Intelligence Engine",
    page_icon="🔮",
    layout="wide",
)

# Load System Fonts & CSS
def load_design_system():
    with open("style.css") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

try:
    load_design_system()
except:
    st.info("Style engine loading...")

# --- Intelligence Config ---
SKILLS_DB = {
    "Engineering": ["Python", "Java", "C++", "C#", "Go", "Rust", "JavaScript", "TypeScript"],
    "Web/Frontend": ["React", "Angular", "Vue", "HTML", "CSS", "TailwindCSS", "Next.js"],
    "Data/AI": ["Machine Learning", "Deep Learning", "NLP", "Pandas", "PyTorch", "TensorFlow", "scikit-learn", "SQL"],
    "Cloud/DevOps": ["AWS", "Azure", "GCP", "Docker", "Kubernetes", "CI/CD", "Terraform"],
    "Soft Skills": ["Leadership", "Communication", "Management", "Agile", "Scrum", "Problem Solving"]
}

# --- Core Logic ---

def extract_pdf_data(file):
    text = ""
    try:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    except Exception as e:
        return f"Error: {e}"
    return text

def calculate_match_score(resume, jd):
    if HAS_AI:
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = model.encode([resume, jd])
            score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return round(score * 100, 2)
        except:
            pass
    
    # Fallback keyword match if AI is missing or fails
    r_words = set(re.findall(r'\w+', resume.lower()))
    j_words = set(re.findall(r'\w+', jd.lower()))
    return round((len(r_words & j_words) / len(j_words)) * 100, 2) if j_words else 0

def deep_analyze_skills(text):
    found = {}
    text_lower = text.lower()
    for category, skills in SKILLS_DB.items():
        found[category] = [s for s in skills if re.search(r'\b' + re.escape(s.lower()) + r'\b', text_lower)]
    return found

# --- UI Components ---

def draw_metric(label, value, subtitle):
    st.markdown(f"""
    <div class="metric-box">
        <p style="color: #94a3b8; font-size: 0.85rem; margin-bottom: 2px;">{label}</p>
        <p style="color: #6366f1; font-size: 2.5rem; font-weight: 800; margin: 0;">{value}</p>
        <p style="color: #64748b; font-size: 0.75rem;">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)

def draw_skill_pills(skills, is_missing=False):
    css_class = "skill-tag skill-tag-missing" if is_missing else "skill-tag"
    html = '<div style="margin-top: 10px;">'
    for s in skills:
        html += f'<span class="{css_class}">{s}</span>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

def main():
    # 1. State Initialization
    keys = ['resume_text_input', 'jd_text_input', 'jd_text_input_file']
    for k in keys:
        if k not in st.session_state:
            st.session_state[k] = ""

    # 2. Sidebar Control
    st.sidebar.markdown("### 🛠️ CONTROL PANEL")
    if st.sidebar.button("🗑️ Reset All Data", use_container_width=True):
        for k in keys: st.session_state[k] = ""
        st.rerun()
    st.sidebar.markdown("---")
    st.sidebar.caption("Resume Engine v2.0 Pro")

    # 3. Hero Section
    st.markdown("""
    <div class="hero-container fade-in">
        <h1 class="hero-text">Resume Engine.</h1>
        <p style="font-size: 1.25rem; color: #94a3b8; max-width: 700px; margin: 0 auto;">
            Next-generation AI analysis for your professional profile. 
            Optimize for ATS, detect skill gaps, and predict career fit.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 4. Input Hub
    _, center, _ = st.columns([1, 4, 1])
    with center:
        st.markdown('<div class="custom-card input-section fade-in">', unsafe_allow_html=True)
        mode = st.radio("Choose Input Method", ["⚡ Paste Content", "📄 Upload Files"], horizontal=True)
        cin1, cin2 = st.columns(2)
        
        # Get base directory for files
        base_dir = os.path.dirname(os.path.abspath(__file__))

        if mode == "📄 Upload Files":
            with cin1:
                f = st.file_uploader("Drop Resume (PDF)", type=["pdf"])
                r_text = extract_pdf_data(f) if f else ""
            with cin2:
                st.text_area("Target Job Description", key="jd_text_input_file", height=180)
            final_r, final_j = r_text, st.session_state.get('jd_text_input_file', '')
        else:
            with cin1:
                st.text_area("Your Resume", key="resume_text_input", height=300)
            with cin2:
                st.text_area("Job Description", key="jd_text_input", height=300)
            final_r, final_j = st.session_state.get('resume_text_input', ''), st.session_state.get('jd_text_input', '')

        st.markdown("</div>", unsafe_allow_html=True)
        
        # Sample Data Button (In middle) - Show if fields are essentially empty
        if not final_r.strip() and not final_j.strip():
            if st.button("✨ QUICK-LOAD SAMPLE PROFILE", use_container_width=True):
                try:
                    jd_path = os.path.join(base_dir, "dummy_jd.txt")
                    res_path = os.path.join(base_dir, "sample_resume_content.txt")
                    
                    with open(jd_path, "r") as f_jd: jd = f_jd.read()
                    with open(res_path, "r") as f_res: res = f_res.read()
                    
                    st.session_state['jd_text_input'] = jd
                    st.session_state['jd_text_input_file'] = jd
                    st.session_state['resume_text_input'] = res
                    st.rerun()
                except Exception as e:
                    st.error(f"Error loading sample data: {e}")

        analyze_btn = st.button("🔥 START INTELLIGENCE SCAN", use_container_width=True)

    # 5. Analysis Hub
    if analyze_btn:
        if not final_r or not final_j:
            st.warning("Action required: Please provide both resume and job description.")
            return

        with st.spinner("Decoding Professional DNA..."):
            score = calculate_match_score(final_r, final_j)
            r_skills = [s for sub in deep_analyze_skills(final_r).values() for s in sub]
            j_skills = [s for sub in deep_analyze_skills(final_j).values() for s in sub]
            matching = sorted(list(set(r_skills) & set(j_skills)))
            missing = sorted(list(set(j_skills) - set(r_skills)))
            
            st.markdown('<div class="custom-card fade-in">', unsafe_allow_html=True)
            m1, m2, m3, m4 = st.columns(4)
            with m1: draw_metric("MATCH SCORE", f"{score}%", "AI Semantic Match")
            with m2: draw_metric("TOTAL SKILLS", len(r_skills), "Competencies Found")
            with m3: draw_metric("ATS READINESS", "HIGH" if score > 75 else "MEDIUM", "Format Compliance")
            with m4: draw_metric("READ TIME", f"{max(1, len(final_r.split())//200)}m", "Industry Standard")
            st.markdown('</div>', unsafe_allow_html=True)

            t1, t2, t3 = st.tabs(["🎯 STRATEGIC GAPS", "📊 KEYWORD ENGINE", "📈 PROFILE HEALTH"])
            with t1:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("### ✅ Strengths")
                    if matching: draw_skill_pills(matching)
                    else: st.info("No matching keywords found.")
                with c2:
                    st.markdown("### 🧨 Critical Gaps")
                    if missing: draw_skill_pills(missing, True)
                    else: st.success("Zero gaps detected!")

            with t2:
                words = Counter([w for w in re.findall(r'\b\w{4,}\b', final_r.lower()) if w not in {'from','with','that','this','have'}]).most_common(8)
                for w, c in words:
                    st.markdown(f"**{w.upper()}** ({c}x)")
                    st.markdown(f'<div class="bar-container"><div class="bar-fill" style="width: {min(100, c*15)}%;"></div></div>', unsafe_allow_html=True)

            with t3:
                checks = {"Contact info": "@" in final_r, "Education": bool(re.search(r'Education|Degree', final_r, re.I)), "Experience": bool(re.search(r'20\d\d|Experience', final_r, re.I)), "Skills": bool(r_skills)}
                for k, v in checks.items():
                    st.markdown(f"<p style='color: {'#10b981' if v else '#ef4444'}; font-size: 1.1rem;'>{'◈' if v else '◇'} {k} detected</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
