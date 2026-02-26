
import time

def mock_analysis():
    print("Loading AI Model (Mocking)...")
    time.sleep(1)
    print("Reading Candidate Resume... [DONE]")
    time.sleep(0.5)
    print("Reading Job Description... [DONE]")
    time.sleep(0.5)
    print("Calculating Semantic Similarity... [DONE]")
    time.sleep(0.5)
    print("Extracting Skills & Gaps... [DONE]")
    
    print("\n" + "="*40)
    print("       RESUME ANALYSIS REPORT       ")
    print("="*40)
    
    print("\n📊 MATCH SCORE: 82%")
    print("   (Good match, but some key skills missing)")
    
    print("\n✅ STRONG AREAS:")
    print("   ✔ Python (Found in resume and JD)")
    print("   ✔ Flask  (Found in resume and JD)")
    print("   ✔ Machine Learning (Bonus skill)")

    print("\n❌ MISSING SKILLS (Critical):")
    print("   ✘ SQL    (Required but not found)")
    print("   ✘ Docker (Required but not found)")
    print("   ✘ AWS    (Required but not found)")

    print("\n⏳ EXPERIENCE GAP:")
    print("   - Required: 3+ years")
    print("   - Found:    2 years")
    print("   -> Gap:     1 year shortfall")
    
    print("\n" + "="*40)

if __name__ == "__main__":
    mock_analysis()
