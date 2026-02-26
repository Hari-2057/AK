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
    # Hero Section
    st.markdown("""
    <div class="hero-container fade-in">
        <h1 class="hero-text">Resume Engine.</h1>
        <p style="font-size: 1.25rem; color: #94a3b8; max-width: 700px; margin: 0 auto;">
            Next-generation AI analysis for your professional profile. 
            Optimize for ATS, detect skill gaps, and predict career fit.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Centered Input Hub
    _, center, _ = st.columns([1, 4, 1])
    
    with center:
        st.markdown('<div class="custom-card input-section fade-in">', unsafe_allow_html=True)
        
        mode = st.radio("Choose Input Method", ["⚡ Paste Content", "📄 Upload Files"], horizontal=True)
        
        col_in1, col_in2 = st.columns(2)
        
        if mode == "📄 Upload Files":
            with col_in1:
                resume_file = st.file_uploader("Drop Resume (PDF)", type=["pdf"])
                resume_text_file = extract_pdf_data(resume_file) if resume_file else ""
            with col_in2:
                jd_text_area = st.text_area("Target Job Description", key="jd_text_input_file", height=180, placeholder="Paste JD here...")
        else:
            with col_in1:
                st.text_area("Your Resume Content", key="resume_text_input", height=300, placeholder="Paste resume text or use 'Try Sample' in sidebar...")
            with col_in2:
                st.text_area("Target Job Description", key="jd_text_input", height=300, placeholder="Paste job description text...")

        st.markdown("</div>", unsafe_allow_html=True)
        
        # Action Center
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        analyze_btn = st.button("🚀 INITIATE INTELLIGENCE SCAN")

    # Sidebar Tools
    st.sidebar.markdown("### 🛠️ Developer Tools")
    
    # Initialize session state for text areas if not already there
    if 'resume_text_input' not in st.session_state:
        st.session_state['resume_text_input'] = ""
    if 'jd_text_input' not in st.session_state:
        st.session_state['jd_text_input'] = ""

    if st.sidebar.button("💡 Try Sample Data"):
        try:
            with open("dummy_jd.txt") as f: 
                st.session_state['jd_text_input'] = f.read()
            with open("sample_resume_content.txt") as f: 
                st.session_state['resume_text_input'] = f.read()
            st.rerun()
        except Exception as e: 
            st.sidebar.error(f"Sample files missing: {e}")

    if st.sidebar.button("🧹 Clear All"):
        st.session_state['resume_text_input'] = ""
        st.session_state['jd_text_input'] = ""
        st.session_state['jd_text_input_file'] = "" # Clear for file mode too
        st.rerun()

    # Final text selection based on mode
    if mode == "📄 Upload Files":
        final_resume_text = resume_text_file
        final_jd_text = st.session_state.get('jd_text_input_file', "")
    else:
        final_resume_text = st.session_state.get('resume_text_input', "")
        final_jd_text = st.session_state.get('jd_text_input', "")

    # --- Analysis Phase ---
    if analyze_btn:
        if not final_resume_text or not final_jd_text:
            st.warning("Action required: Please provide both resume and job description.")
            return

        with st.spinner("Decoding Professional DNA..."):
            # 1. Main Scores
            score = calculate_match_score(final_resume_text, final_jd_text)
            
            # 2. Skills Analysis
            r_skills_all = [s for sub in deep_analyze_skills(final_resume_text).values() for s in sub]
            j_skills_all = [s for sub in deep_analyze_skills(final_jd_text).values() for s in sub]
            
            matching = sorted(list(set(r_skills_all) & set(j_skills_all)))
            missing = sorted(list(set(j_skills_all) - set(r_skills_all)))
            
            # 3. Enhanced Features
            word_count = len(final_resume_text.split())
            read_time = max(1, word_count // 200)
            
            # Result Visualization
            st.markdown('<div class="custom-card fade-in">', unsafe_allow_html=True)
            m1, m2, m3, m4 = st.columns(4)
            with m1: draw_metric("MATCH SCORE", f"{score}%", "AI Semantic Match")
            with m2: draw_metric("TOTAL SKILLS", len(r_skills_all), "Detected Competencies")
            with m3: draw_metric("ATS READINESS", "HIGH" if score > 75 else "MEDIUM", "Format Compliance")
            with m4: draw_metric("READ TIME", f"{read_time}m", f"{word_count} Words")
            st.markdown('</div>', unsafe_allow_html=True)

            # Details Section
            t1, t2, t3 = st.tabs(["🎯 STRATEGIC GAPS", "� KEYWORD ENGINE", "� PROFILE HEALTH"])
            
            with t1:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("### ✅ Strengths")
                    if matching: draw_skill_pills(matching)
                    else: st.info("No matching database keywords found.")
                with c2:
                    st.markdown("### 🧨 Missing Criticals")
                    if missing: draw_skill_pills(missing, is_missing=True)
                    else: st.success("Zero keyword gaps detected!")

            with t2:
                st.markdown("### Resume Keyword Density")
                words = re.findall(r'\b\w{4,}\b', final_resume_text.lower())
                common = Counter([w for w in words if w not in {'from', 'with', 'that', 'this', 'have'}]).most_common(8)
                
                for word, count in common:
                    st.markdown(f"**{word.upper()}** ({count}x)")
                    st.markdown(f"""
                    <div class="bar-container">
                        <div class="bar-fill" style="width: {min(100, count*15)}%;"></div>
                    </div>
                    """, unsafe_allow_html=True)

            with t3:
                st.markdown("### Professional Format Consistency")
                checks = {
                    "Contact Info Found": bool(re.search(r'@', final_resume_text)),
                    "Education Section": bool(re.search(r'Education|University|Degree', final_resume_text, re.I)),
                    "Experience Timeline": bool(re.search(r'20\d\d|Experience|History', final_resume_text, re.I)),
                    "Skills Highlighted": bool(r_skills_all)
                }
                for check, status in checks.items():
                    color = "#10b981" if status else "#ef4444"
                    icon = "◈" if status else "◇"
                    st.markdown(f"<p style='color: {color}; font-size: 1.1rem; font-weight: 500;'>{icon} {check}</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
