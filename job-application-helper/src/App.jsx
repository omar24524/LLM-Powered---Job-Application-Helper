import { useState, useRef, useCallback } from 'react'
import * as pdfjsLib from 'pdfjs-dist'
import mammoth from 'mammoth'
import './App.css'

// pdfjs worker
pdfjsLib.GlobalWorkerOptions.workerSrc = new URL(
  'pdfjs-dist/build/pdf.worker.min.mjs',
  import.meta.url
).toString()

// ─── Constants ────────────────────────────────────────────────────────────────
const AGENTS_META = [
  { id: 'cv',    label: 'Tailored CV',    icon: 'cv'    },
  { id: 'cover', label: 'Cover letter',   icon: 'cover' },
  { id: 'email', label: 'Follow-up email',icon: 'email' },
]

const PIPELINE_IDS = ['orchestrator', 'cv', 'cover', 'email', 'critic']

const LLM_MODES = [
  { id: 'local', label: 'Local (Phi)' },
  { id: 'cloud', label: 'Anthropic API' },
]

// ─── File parsing ─────────────────────────────────────────────────────────────
async function parsePDF(file) {
  const buffer = await file.arrayBuffer()
  const pdf = await pdfjsLib.getDocument({ data: buffer }).promise
  let text = ''
  for (let i = 1; i <= pdf.numPages; i++) {
    const page = await pdf.getPage(i)
    const content = await page.getTextContent()
    text += content.items.map(item => item.str).join(' ') + '\n'
  }
  return text.trim()
}

async function parseDOCX(file) {
  const buffer = await file.arrayBuffer()
  const result = await mammoth.extractRawText({ arrayBuffer: buffer })
  return result.value.trim()
}

async function parseFile(file) {
  const ext = file.name.split('.').pop().toLowerCase()
  if (ext === 'pdf') return parsePDF(file)
  if (ext === 'docx') return parseDOCX(file)
  throw new Error('Unsupported file type. Use PDF or DOCX.')
}

// ─── LLM calls ───────────────────────────────────────────────────────────────
async function callLocal(systemPrompt, userMessage) {
  const res = await fetch('/llm/v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: 'phi',
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user',   content: userMessage  },
      ],
      max_tokens: 1200,
      temperature: 0.7,
      stream: false,
    }),
  })
  if (!res.ok) throw new Error(`Local LLM error: ${res.status} ${res.statusText}`)
  const data = await res.json()
  return data.choices?.[0]?.message?.content || ''
}

async function streamCloud(apiKey, systemPrompt, userMessage, onChunk) {
  const res = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': apiKey,
      'anthropic-version': '2023-06-01',
      'anthropic-dangerous-direct-browser-access': 'true',
    },
    body: JSON.stringify({
      model: 'claude-sonnet-4-6',
      max_tokens: 1200,
      stream: true,
      system: systemPrompt,
      messages: [{ role: 'user', content: userMessage }],
    }),
  })
  if (!res.ok) { const e = await res.json(); throw new Error(e.error?.message || 'API error') }
  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let full = ''
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    for (const line of decoder.decode(value).split('\n')) {
      if (!line.startsWith('data: ')) continue
      const d = line.slice(6).trim()
      if (d === '[DONE]') continue
      try {
        const delta = JSON.parse(d)?.delta?.text || ''
        if (delta) { full += delta; onChunk(full) }
      } catch {}
    }
  }
  return full
}

async function callCloud(apiKey, systemPrompt, userMessage) {
  const res = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': apiKey,
      'anthropic-version': '2023-06-01',
      'anthropic-dangerous-direct-browser-access': 'true',
    },
    body: JSON.stringify({
      model: 'claude-sonnet-4-6',
      max_tokens: 1000,
      system: systemPrompt,
      messages: [{ role: 'user', content: userMessage }],
    }),
  })
  const data = await res.json()
  if (data.error) throw new Error(data.error.message)
  return data.content.map(b => b.text || '').join('')
}

// ─── Agent configs ────────────────────────────────────────────────────────────
const ORCHESTRATOR_SYSTEM = `You are the Orchestrator agent in a job application pipeline. Analyze the CV and job description and return a compact JSON object (no markdown, raw JSON only) with:
- "key_requirements": array of top 5 requirements from the JD
- "cv_gaps": array of up to 3 skills the CV should emphasize more
- "tone": one word for company culture ("formal", "startup", "corporate", "creative")
- "role_title": exact job title from the JD
- "company_name": company name if mentioned, else "the company"
Return ONLY the JSON object, no other text.`

const CRITIC_SYSTEM = `You are a strict hiring expert reviewing AI-generated job application materials.
Evaluate each piece and return raw JSON only (no markdown) with this structure:
{"cv":{"score":1-10,"verdict":"one sentence","issues":["issue1"]},"cover":{"score":1-10,"verdict":"one sentence","issues":["issue1"]},"email":{"score":1-10,"verdict":"one sentence","issues":["issue1"]}}
Score 8+ means publish-ready. Be specific. Return ONLY JSON.`

function buildAgentConfigs(cv, jd, brief) {
  const req = (brief.key_requirements || []).join(', ')
  const gaps = (brief.cv_gaps || []).join(', ')
  return [
    {
      id: 'cv',
      system: `You are a professional CV writer. Rewrite and tailor the candidate's CV to better match the role. Sharpen bullet points, emphasize relevant skills, weave in keywords from requirements. Use plain text only (no markdown headers). Be concise and impactful.`,
      user: `Original CV:\n${cv}\n\nRole: ${brief.role_title} at ${brief.company_name}\nKey requirements: ${req}\nAreas to strengthen: ${gaps}\n\nRewrite the CV tailored for this role.`,
    },
    {
      id: 'cover',
      system: `You are an expert cover letter writer. Write a compelling, personalized cover letter. Tone: ${brief.tone}. 3 paragraphs: hook + fit + closing. No generic filler. Address to the hiring manager.`,
      user: `CV:\n${cv}\n\nRole: ${brief.role_title} at ${brief.company_name}\nKey requirements: ${req}\n\nWrite the cover letter.`,
    },
    {
      id: 'email',
      system: `You are a professional email writer. Draft a short, confident follow-up email to send 5 days after submitting a job application. Tone: ${brief.tone}. Under 100 words. Include subject line. Professional interest only — no desperation.`,
      user: `Role: ${brief.role_title} at ${brief.company_name}\nApplicant summary:\n${cv.substring(0, 400)}\n\nWrite the follow-up email.`,
    },
  ]
}

// ─── Icon components ──────────────────────────────────────────────────────────
function Icon({ name, size = 16 }) {
  const icons = {
    cv: <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>,
    cover: <path strokeLinecap="round" strokeLinejoin="round" d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"/>,
    email: <path strokeLinecap="round" strokeLinejoin="round" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"/>,
    upload: <><path strokeLinecap="round" strokeLinejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"/></>,
    copy: <path strokeLinecap="round" strokeLinejoin="round" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"/>,
    refresh: <path strokeLinecap="round" strokeLinejoin="round" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"/>,
    check: <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7"/>,
    close: <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12"/>,
    file: <path strokeLinecap="round" strokeLinejoin="round" d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z"/>,
    play: <path strokeLinecap="round" strokeLinejoin="round" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"/>,
    cpu: <><rect x="4" y="4" width="16" height="16" rx="2" ry="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/></>,
    cloud: <path strokeLinecap="round" strokeLinejoin="round" d="M3 15a4 4 0 004 4h9a5 5 0 10-.1-9.999 5.002 5.002 0 10-9.78 2.096A4.001 4.001 0 003 15z"/>,
    warning: <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/>,
  }
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" aria-hidden="true">
      {icons[name]}
    </svg>
  )
}

// ─── File drop zone ───────────────────────────────────────────────────────────
function FileDropZone({ onText, label }) {
  const [dragging, setDragging] = useState(false)
  const [fileName, setFileName] = useState('')
  const [parsing, setParsing] = useState(false)
  const [error, setError] = useState('')
  const inputRef = useRef()

  async function handleFile(file) {
    if (!file) return
    setError('')
    setParsing(true)
    setFileName(file.name)
    try {
      const text = await parseFile(file)
      onText(text)
    } catch (e) {
      setError(e.message)
      setFileName('')
    } finally {
      setParsing(false)
    }
  }

  const onDrop = useCallback((e) => {
    e.preventDefault()
    setDragging(false)
    handleFile(e.dataTransfer.files[0])
  }, [])

  return (
    <div
      className={`dropzone ${dragging ? 'dragging' : ''} ${fileName ? 'has-file' : ''}`}
      onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
      onClick={() => inputRef.current.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".pdf,.docx"
        style={{ display: 'none' }}
        onChange={(e) => handleFile(e.target.files[0])}
      />
      {parsing ? (
        <span className="dz-status parsing">Parsing {fileName}…</span>
      ) : fileName ? (
        <span className="dz-status has-file">
          <Icon name="file" size={14} /> {fileName}
          <button className="dz-clear" onClick={(e) => { e.stopPropagation(); setFileName(''); onText('') }}>
            <Icon name="close" size={12} />
          </button>
        </span>
      ) : (
        <span className="dz-status idle">
          <Icon name="upload" size={14} /> {label} — PDF or DOCX
        </span>
      )}
      {error && <span className="dz-error">{error}</span>}
    </div>
  )
}

// ─── Score badge ──────────────────────────────────────────────────────────────
function ScoreBadge({ score }) {
  if (!score) return null
  const cls = score >= 8 ? 'good' : score >= 6 ? 'mid' : 'low'
  return <span className={`score-badge ${cls}`}>{score}/10</span>
}

// ─── Pill ─────────────────────────────────────────────────────────────────────
function Pill({ id, state, label }) {
  return (
    <span className={`pill ${state || ''}`}>
      <span className={`dot ${state === 'active' ? 'dot-active' : state === 'done' ? 'dot-done' : state === 'error' ? 'dot-error' : ''}`} />
      {label}
    </span>
  )
}

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const [mode, setMode]             = useState('local')
  const [apiKey, setApiKey]         = useState('')
  const [cv, setCv]                 = useState('')
  const [jd, setJd]                 = useState('')
  const [running, setRunning]       = useState(false)
  const [agentState, setAgentState] = useState({})
  const [outputs, setOutputs]       = useState({})
  const [critique, setCritique]     = useState(null)
  const [copied, setCopied]         = useState({})
  const [rewriting, setRewriting]   = useState({})
  const [localError, setLocalError] = useState('')
  const briefRef = useRef(null)
  const outputsRef = useRef({})

  function setAgent(id, state) { setAgentState(prev => ({ ...prev, [id]: state })) }

  function setOutput(id, text) {
    outputsRef.current = { ...outputsRef.current, [id]: text }
    setOutputs(prev => ({ ...prev, [id]: text }))
  }

  // ── Universal call + stream ─────────────────────────────────────────────
  async function call(sys, usr) {
    if (mode === 'local') return callLocal(sys, usr)
    return callCloud(apiKey, sys, usr)
  }

  async function stream(sys, usr, onChunk) {
    if (mode === 'local') {
      // local LLM: no streaming, simulate with single call
      const result = await callLocal(sys, usr)
      onChunk(result)
      return result
    }
    return streamCloud(apiKey, sys, usr, onChunk)
  }

  // ── Critic ──────────────────────────────────────────────────────────────
  async function runCritic(brief) {
    setAgent('critic', 'active')
    const snap = outputsRef.current
    try {
      const raw = await call(
        CRITIC_SYSTEM,
        `Role: ${brief?.role_title} at ${brief?.company_name}
Requirements: ${(brief?.key_requirements || []).join(', ')}

--- TAILORED CV ---\n${snap['cv'] || '(none)'}
--- COVER LETTER ---\n${snap['cover'] || '(none)'}
--- FOLLOW-UP EMAIL ---\n${snap['email'] || '(none)'}`
      )
      const parsed = JSON.parse(raw.replace(/```json|```/g, '').trim())
      setCritique(parsed)
      setAgent('critic', 'done')
    } catch {
      setAgent('critic', 'error')
    }
  }

  // ── Full pipeline ────────────────────────────────────────────────────────
  async function runPipeline() {
    if (mode === 'cloud' && !apiKey.trim()) { alert('Enter your Anthropic API key.'); return }
    if (!cv.trim() || !jd.trim()) { alert('Provide both CV content and a job description.'); return }

    setRunning(true)
    setLocalError('')
    setAgentState({})
    setOutputs({})
    outputsRef.current = {}
    setCritique(null)

    // Orchestrator
    setAgent('orchestrator', 'active')
    let brief
    try {
      const raw = await call(ORCHESTRATOR_SYSTEM, `CV:\n${cv}\n\nJob Description:\n${jd}`)
      brief = JSON.parse(raw.replace(/```json|```/g, '').trim())
      briefRef.current = brief
      setAgent('orchestrator', 'done')
    } catch (e) {
      setAgent('orchestrator', 'error')
      if (mode === 'local') setLocalError('Could not reach local LLM. Is llama-server running on port 8080?')
      setRunning(false)
      return
    }

    // Three content agents
    const configs = buildAgentConfigs(cv, jd, brief)
    for (const agent of configs) {
      setAgent(agent.id, 'active')
      try {
        await stream(agent.system, agent.user, (partial) => setOutput(agent.id, partial))
        setAgent(agent.id, 'done')
      } catch (e) {
        setOutput(agent.id, `Error: ${e.message}`)
        setAgent(agent.id, 'error')
      }
    }

    await runCritic(brief)
    setRunning(false)
  }

  // ── Regenerate single ────────────────────────────────────────────────────
  async function regenerate(id) {
    const brief = briefRef.current
    if (!brief) return
    setRewriting(prev => ({ ...prev, [id]: true }))
    setAgent(id, 'active')
    setOutput(id, '')
    const config = buildAgentConfigs(cv, jd, brief).find(c => c.id === id)
    try {
      await stream(config.system, config.user, (partial) => setOutput(id, partial))
      setAgent(id, 'done')
    } catch (e) {
      setOutput(id, `Error: ${e.message}`)
      setAgent(id, 'error')
    }
    setRewriting(prev => ({ ...prev, [id]: false }))
    await runCritic(brief)
  }

  function copyText(id) {
    const text = outputsRef.current[id] || outputs[id]
    if (!text) return
    navigator.clipboard.writeText(text).then(() => {
      setCopied(prev => ({ ...prev, [id]: true }))
      setTimeout(() => setCopied(prev => ({ ...prev, [id]: false })), 2000)
    })
  }

  const hasStarted = Object.keys(agentState).length > 0

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-title">
          <h1>Job Application Helper</h1>
          <span className="header-tag">Multi-agent · Phase 3</span>
        </div>

        {/* Mode switcher */}
        <div className="mode-switcher">
          {LLM_MODES.map(m => (
            <button
              key={m.id}
              className={`mode-btn ${mode === m.id ? 'active' : ''}`}
              onClick={() => setMode(m.id)}
              disabled={running}
            >
              <Icon name={m.id === 'local' ? 'cpu' : 'cloud'} size={13} />
              {m.label}
            </button>
          ))}
        </div>
      </header>

      {/* Cloud API key */}
      {mode === 'cloud' && (
        <div className="api-key-bar">
          <label>API key</label>
          <input
            type="password"
            placeholder="sk-ant-..."
            value={apiKey}
            onChange={e => setApiKey(e.target.value)}
          />
          <span className="hint">Stored in memory only.</span>
        </div>
      )}

      {/* Local LLM info */}
      {mode === 'local' && (
        <div className="info-bar">
          <Icon name="cpu" size={14} />
          <span>Start llama-server before running: <code>llama-server -m your-phi-model.gguf --port 8080</code></span>
        </div>
      )}

      {localError && (
        <div className="error-bar">
          <Icon name="warning" size={14} />
          {localError}
        </div>
      )}

      {/* Inputs */}
      <div className="inputs-section">
        <div className="input-col">
          <div className="input-label">Your CV</div>
          <FileDropZone label="Upload CV" onText={setCv} />
          <textarea
            className="input-textarea"
            placeholder="…or paste CV text here"
            value={cv}
            onChange={e => setCv(e.target.value)}
            rows={8}
          />
        </div>
        <div className="input-col">
          <div className="input-label">Job description</div>
          <textarea
            className="input-textarea jd-textarea"
            placeholder="Paste the full job description here…"
            value={jd}
            onChange={e => setJd(e.target.value)}
            rows={10}
          />
        </div>
      </div>

      {/* Run button */}
      <button className="run-btn" onClick={runPipeline} disabled={running}>
        {running
          ? <><span className="spinner" />Running pipeline…</>
          : <><Icon name="play" size={15} />{hasStarted ? 'Run again' : 'Run pipeline'}</>
        }
      </button>

      {/* Pipeline tracker */}
      {hasStarted && (
        <div className="pipeline">
          {PIPELINE_IDS.map((id, i) => (
            <span key={id} style={{ display: 'inline-flex', alignItems: 'center' }}>
              <Pill
                id={id}
                state={agentState[id]}
                label={
                  id === 'orchestrator' ? 'Orchestrator'
                  : id === 'critic' ? 'Critic'
                  : AGENTS_META.find(a => a.id === id)?.label
                }
              />
              {i < PIPELINE_IDS.length - 1 && <span className="pipe-arrow">›</span>}
            </span>
          ))}
        </div>
      )}

      {/* Output cards */}
      {hasStarted && (
        <div className="outputs">
          {AGENTS_META.map(agent => {
            const text = outputs[agent.id]
            const state = agentState[agent.id]
            const crit = critique?.[agent.id]
            const isRw = rewriting[agent.id]
            if (!state) return null

            return (
              <div className="output-card" key={agent.id}>
                <div className="card-header">
                  <div className="card-left">
                    <Icon name={agent.icon} size={15} />
                    <span className="card-title">{agent.label}</span>
                    <ScoreBadge score={crit?.score} />
                  </div>
                  <div className="card-right">
                    {text && !isRw && (
                      <>
                        <button
                          className="icon-btn"
                          onClick={() => regenerate(agent.id)}
                          disabled={running || isRw}
                          title="Regenerate"
                        >
                          <Icon name="refresh" size={14} />
                          Regenerate
                        </button>
                        <button className="icon-btn" onClick={() => copyText(agent.id)} title="Copy">
                          <Icon name={copied[agent.id] ? 'check' : 'copy'} size={14} />
                          {copied[agent.id] ? 'Copied' : 'Copy'}
                        </button>
                      </>
                    )}
                  </div>
                </div>

                <div className="card-body">
                  {(state === 'active' || isRw) && !text && (
                    <div className="skeleton-wrap">
                      {[85, 70, 92, 60, 78].map((w, i) => (
                        <div key={i} className="skeleton" style={{ width: `${w}%` }} />
                      ))}
                    </div>
                  )}
                  {text && (
                    <pre className={(state === 'active' || isRw) ? 'streaming' : ''}>{text}</pre>
                  )}
                </div>

                {crit && (
                  <div className="critic-panel">
                    <p className="critic-verdict">"{crit.verdict}"</p>
                    {crit.issues?.length > 0 && (
                      <ul className="critic-issues">
                        {crit.issues.map((issue, i) => <li key={i}>{issue}</li>)}
                      </ul>
                    )}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
