# Turkish Fact-Check Assistant (RAG)

A retrieval-augmented generation (RAG) system that checks Turkish claims against
an archive of professional fact-checks and returns a sourced verdict.

> Work in progress — capstone project for the Workintech Data Science & AI program.

## Data
- **FCTR** (Cekinel et al., LREC-COLING 2024): 3,238 Turkish fact-checked claims
  from Teyit, Doğruluk Payı and Doğrula (2016–2023).
  Download from https://github.com/firatcekinel/FCTR and place `fctr.csv` in `data/raw/`.
- **Google Fact Check Tools API**: recent fact-checks from the same publishers.

## Setup
```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```
Create a `.env` file in the project root:
```
GEMINI_API_KEY=your_key
FACTCHECK_API_KEY=your_key
```
