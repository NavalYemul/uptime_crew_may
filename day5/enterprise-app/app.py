from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from rag_pipeline import ask_question, get_pipeline_status

app = FastAPI(title="Enterprise RAG Agent API")

_ROOT_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Enterprise RAG Agent API</title>
  <style>
    :root {
      --bg: #f5efe4;
      --card: rgba(255, 250, 241, 0.92);
      --ink: #1f2a2a;
      --muted: #5d6b66;
      --accent: #0f766e;
      --warm: #c96f3b;
      --border: rgba(31, 42, 42, 0.12);
      --shadow: 0 20px 50px rgba(62, 40, 15, 0.08);
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: ui-sans-serif, system-ui, sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(201, 111, 59, 0.18), transparent 28%),
        radial-gradient(circle at top right, rgba(15, 118, 110, 0.16), transparent 22%),
        linear-gradient(180deg, #f8f2e8 0%, var(--bg) 48%, #efe6d7 100%);
    }
    main {
      max-width: 980px;
      margin: 0 auto;
      padding: 40px 20px 56px;
    }
    .hero, .panel {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 24px;
      box-shadow: var(--shadow);
    }
    .hero {
      padding: 28px;
      margin-bottom: 18px;
    }
    .eyebrow {
      text-transform: uppercase;
      letter-spacing: 0.12em;
      font-size: 12px;
      font-weight: 700;
      color: var(--warm);
      margin-bottom: 10px;
    }
    h1 {
      margin: 0 0 12px;
      font-size: 42px;
      line-height: 1;
    }
    p {
      color: var(--muted);
      line-height: 1.6;
    }
    .pill-row {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 18px;
    }
    .pill {
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(255,255,255,0.9);
      border: 1px solid var(--border);
      font-size: 14px;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
      margin: 18px 0;
    }
    .panel {
      padding: 20px;
    }
    .label {
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      margin-bottom: 8px;
    }
    .value {
      font-size: 26px;
      font-weight: 800;
      margin-bottom: 8px;
    }
    code, pre {
      background: rgba(255,255,255,0.88);
      border-radius: 12px;
      border: 1px solid var(--border);
    }
    code {
      padding: 2px 6px;
    }
    pre {
      padding: 14px;
      overflow: auto;
      white-space: pre-wrap;
    }
    a { color: var(--accent); text-decoration: none; }
    ul { padding-left: 20px; }
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <div class="eyebrow">Local-First Demo API</div>
      <h1>Enterprise RAG Agent</h1>
      <p>
        This backend answers from local files first and lets the model call a free
        <code>web_search</code> tool only when the local corpus is weak or missing the answer.
      </p>
      <div class="pill-row">
        <span class="pill">FastAPI backend</span>
        <span class="pill">Local files from <code>data/</code></span>
        <span class="pill">Tool-calling web fallback</span>
      </div>
    </section>

    <section class="grid">
      <div class="panel">
        <div class="label">Explore</div>
        <div class="value">API Docs</div>
        <p>Try requests directly in the browser.</p>
        <ul>
          <li><a href="/docs">Swagger UI</a></li>
          <li><a href="/redoc">ReDoc</a></li>
          <li><a href="/openapi.json">OpenAPI JSON</a></li>
          <li><a href="/status">Pipeline status</a></li>
        </ul>
      </div>
      <div class="panel">
        <div class="label">Run It</div>
        <div class="value">Quick Start</div>
        <pre>cp .env.example .env
./start_demo.sh</pre>
        <p>If port 8000 is already in use, run <code>BACKEND_PORT=8001 ./start_demo.sh</code>.</p>
      </div>
      <div class="panel">
        <div class="label">Try It</div>
        <div class="value">Sample Request</div>
        <pre>{
  "query": "What is Orbit?"
}</pre>
        <p>Post that body to <code>/ask</code>.</p>
      </div>
    </section>
  </main>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def root():
    return _ROOT_HTML


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/status")
def status():
    return get_pipeline_status()


class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        description="The user question to answer.",
        examples=["What is Orbit?"],
    )


@app.post("/ask")
def ask(req: QueryRequest):
    try:
        return ask_question(req.query)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
