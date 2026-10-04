# Calculus Solver

A real website: Flask backend + SymPy engine + browser frontend. No AI service needed.

## Run on Windows (cmd.exe)
    cd calculus-solver
    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt
    python app.py
Open http://localhost:5000

## Deploy free on Render
1. Push this folder to a GitHub repo.
2. Render.com > New > Web Service > connect the repo.
3. Build command: `pip install -r requirements.txt`  Start command: `gunicorn app:app --timeout 60`

## Files
- `solver.py`  parses the question, solves it with SymPy, and builds the steps (derivative rules, integration rules, L'Hopital, Taylor table, ODE type)
- `app.py`     Flask server: `POST /api/solve` with `{"q": "..."}`
- `static/index.html`  the page (KaTeX for maths, canvas for graphs)
