# Job Application Helper

Multi-agent pipeline: CV tailor · Cover letter · Follow-up email · Critic

## Features
- **Local LLM** (Phi mini .gguf via llama-server) or **Anthropic cloud** API
- Upload CV as **PDF or DOCX** (or paste text)
- Streaming responses token-by-token
- Critic agent scores each output and flags issues
- Regenerate any single output without re-running the whole pipeline

## Setup

```bash
npm install
npm run dev
```

Open http://localhost:5173

## Using local Phi model

Install llama.cpp then start the server:

```bash
llama-server -m phi-4-mini-instruct.gguf --port 8080 -c 4096
```

In the app, select **Local (Phi)** in the top-right toggle. No API key needed.

## Using Anthropic cloud

Select **Anthropic API** in the toggle and paste your API key from https://console.anthropic.com

## Project structure

```
src/
  App.jsx   ← all agent logic, file parsing, UI
  App.css   ← styles
vite.config.js  ← proxies /llm → localhost:8080
```
