# AI Resume Matcher

This serves as a prototype for the AI Resume Matching engine.

## Setup
1. Ensure you have Python installed.
2. The dependencies are installed in the `venv` directory.
   ```bash
   source venv/bin/activate
   pip install -r requirements.txt
   ```

## Running the App
To start the application:
```bash
streamlit run app.py
```

## Features
- **Upload Resume**: Supports PDF format.
- **Job Description**: Paste the JD text.
- **Analysis**:
  - Match Score (Cosine Similarity with BERT embeddings)
  - Missing Skills (Gap Analysis)
  - Experience Gap (Heuristic based)

## Sample Data included (Coming soon)
We will generate dummy resumes for testing.
