
import re
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Reuse the logic
SKILLS_DB = {
    "Python", "Java", "C++", "JavaScript", "TypeScript", "HTML", "CSS", "React", "Angular", "Vue",
    "Node.js", "Django", "Flask", "SQL", "PostgreSQL", "MySQL", "MongoDB", "NoSQL",
    "Docker", "Kubernetes", "AWS", "Azure", "GCP", "CI/CD", "Git", "GitHub",
    "Machine Learning", "Deep Learning", "Data Science", "NLP", "Computer Vision",
    "scikit-learn", "pandas", "numpy", "matplotlib", "seaborn", "TensorFlow", "PyTorch",
    "Communication", "Teamwork", "Leadership", "Agile", "Scrum",
    "Excel", "Power BI", "Tableau"
}

def extract_skills(text):
    found_skills = set()
    text_lower = text.lower()
    for skill in SKILLS_DB:
        if re.search(r'\b' + re.escape(skill.lower()) + r'\b', text_lower):
            found_skills.add(skill)
    return found_skills

def calculate_similarity(resume_text, job_desc_text):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode([resume_text, job_desc_text])
    score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return round(score * 100, 2)

def extract_years_experience(text):
    match = re.search(r'(\d+)\+?\s*years?', text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 0

def run_demo():
    print("Loading AI Model...")
    
    # Load dummy data
    try:
        with open("dummy_jd.txt", "r") as f:
            job_desc = f.read()
        with open("sample_resume_content.txt", "r") as f:
            resume_text = f.read()
    except:
        print("Error reading dummy files.")
        return

    print("\n--- Processing Resume ---")
    
    # 1. Match Score
    match_score = calculate_similarity(resume_text, job_desc)
    
    # 2. Skill Extraction
    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_desc)
    
    missing_skills = job_skills - resume_skills
    strong_areas = resume_skills.intersection(job_skills)
    
    # 3. Experience Gap
    resume_exp = extract_years_experience(resume_text)
    job_exp = extract_years_experience(job_desc)
    exp_gap = max(0, job_exp - resume_exp)

    # Output
    print("\n" + "="*30)
    print(f"Match Score: {match_score}%")
    print("="*30)
    
    print("\n✅ Strong Areas:")
    if strong_areas:
        for skill in strong_areas:
            print(f" - {skill}")
    else:
        print(" None found.")

    print("\n❌ Missing Skills:")
    if missing_skills:
        for skill in missing_skills:
            print(f" - {skill}")
    else:
        print(" None.")

    print(f"\n⏳ Experience Gap: {exp_gap} Years" if exp_gap > 0 else "\n⏳ Experience Gap: None")
    print("="*30 + "\n")

if __name__ == "__main__":
    run_demo()
