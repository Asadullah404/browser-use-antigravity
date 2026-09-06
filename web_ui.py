import asyncio
import json
import logging
import os
import time
from typing import Literal

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

load_dotenv()

from browser_use import Agent, Browser
from browser_use.browser.views import BrowserStateSummary
from browser_use.llm.antigravity.chat import ChatAntigravity
from browser_use.llm.browser_use.chat import ChatBrowserUse
from browser_use.llm.google.chat import ChatGoogle
from browser_use.llm.openai.chat import ChatOpenAI

app = FastAPI(title="Browser Use Web UI")

class TaskRequest(BaseModel):
	task: str
	model: Literal['antigravity', 'browser-use', 'google', 'openai'] = 'antigravity'
	api_key: str | None = None
	headless: bool = False
	max_steps: int = 30


class SSEQueueLogHandler(logging.Handler):
	"""Logging handler that pipes log records into an asyncio Queue for SSE streaming."""

	def __init__(self, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop, start_time: float):
		super().__init__()
		self.queue = queue
		self.loop = loop
		self.start_time = start_time

	def emit(self, record: logging.LogRecord):
		try:
			msg = self.format(record)
			elapsed = time.time() - self.start_time
			payload = {
				"type": "log",
				"level": record.levelname,
				"name": record.name,
				"text": msg,
				"elapsed": round(elapsed, 1),
			}
			if self.loop.is_running():
				self.loop.call_soon_threadsafe(self.queue.put_nowait, payload)
		except Exception:
			pass


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Browser Use - Task Control Center</title>
    <style>
        :root {
            --bg: #0b0f19;
            --surface: #151d2e;
            --surface-hover: #1e293b;
            --border: #2a374d;
            --primary: #3b82f6;
            --primary-hover: #2563eb;
            --text: #f8fafc;
            --text-muted: #94a3b8;
            --success: #22c55e;
            --warning: #f59e0b;
            --error: #ef4444;
            --terminal-bg: #070a11;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            padding: 30px 16px;
        }
        .container {
            width: 100%;
            max-width: 860px;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        .header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid var(--border);
            padding-bottom: 16px;
        }
        .header h1 {
            font-size: 22px;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .badge {
            font-size: 12px;
            padding: 4px 8px;
            border-radius: 6px;
            background: #1e3a8a;
            color: #93c5fd;
            font-weight: 600;
        }
        .card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 22px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.25);
        }
        .form-group {
            display: flex;
            flex-direction: column;
            gap: 8px;
            margin-bottom: 18px;
        }
        label {
            font-size: 14px;
            font-weight: 600;
            color: var(--text);
        }
        textarea, select, input {
            background: var(--terminal-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px 14px;
            color: var(--text);
            font-size: 14px;
            outline: none;
            transition: border-color 0.2s;
        }
        textarea:focus, select:focus, input:focus {
            border-color: var(--primary);
        }
        textarea {
            resize: vertical;
            min-height: 110px;
            line-height: 1.5;
        }
        .row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
        }
        .checkbox-group {
            display: flex;
            align-items: center;
            gap: 10px;
            user-select: none;
            cursor: pointer;
            padding-top: 10px;
        }
        .checkbox-group input {
            width: 18px;
            height: 18px;
            cursor: pointer;
        }
        .btn {
            background: var(--primary);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 14px 20px;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            transition: background-color 0.2s, opacity 0.2s;
            width: 100%;
        }
        .btn:hover:not(:disabled) {
            background: var(--primary-hover);
        }
        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }
        .spinner {
            border: 3px solid rgba(255,255,255,0.3);
            border-top: 3px solid white;
            border-radius: 50%;
            width: 18px;
            height: 18px;
            animation: spin 0.8s linear infinite;
            display: none;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        /* Activity Tracker Styles */
        .live-panel {
            display: none;
            border-radius: 12px;
            background: var(--surface);
            border: 1px solid var(--border);
            overflow: hidden;
        }
        .live-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 14px 18px;
            background: #111726;
            border-bottom: 1px solid var(--border);
        }
        .status-tracker {
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 14px;
            font-weight: 600;
        }
        .pulse-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: var(--primary);
            box-shadow: 0 0 8px var(--primary);
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse {
            0% { transform: scale(0.9); opacity: 0.7; }
            50% { transform: scale(1.2); opacity: 1; }
            100% { transform: scale(0.9); opacity: 0.7; }
        }
        .timer {
            font-family: monospace;
            font-size: 13px;
            color: var(--text-muted);
            background: var(--terminal-bg);
            padding: 4px 8px;
            border-radius: 6px;
            border: 1px solid var(--border);
        }
        .terminal-box {
            background: var(--terminal-bg);
            padding: 16px;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-size: 13px;
            line-height: 1.6;
            max-height: 380px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 6px;
        }
        .log-entry {
            display: flex;
            gap: 10px;
            word-break: break-word;
        }
        .log-time {
            color: var(--text-muted);
            user-select: none;
            flex-shrink: 0;
            font-size: 12px;
        }
        .log-body {
            color: #cbd5e1;
        }
        .log-info { color: #93c5fd; }
        .log-warn { color: #fcd34d; }
        .log-error { color: #fca5a5; }
        .log-step { color: #a7f3d0; font-weight: bold; }
        .log-action { color: #fbcfe8; }
        .log-thought { color: #c4b5fd; font-style: italic; }

        /* Output Summary Card */
        .output-card {
            display: none;
        }
        .output-header {
            font-size: 14px;
            font-weight: 600;
            color: var(--text-muted);
            margin-bottom: 12px;
            display: flex;
            justify-content: space-between;
        }
        .output-content {
            background: var(--terminal-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 16px;
            font-size: 14px;
            line-height: 1.6;
            white-space: pre-wrap;
            word-break: break-word;
            max-height: 400px;
            overflow-y: auto;
        }
        .status-badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
        }
        .badge-running { background: #1e3a8a; color: #93c5fd; }
        .badge-success { background: #14532d; color: #86efac; }
        .badge-error { background: #7f1d1d; color: #fca5a5; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌐 Browser Use UI</h1>
            <span class="badge">Real-Time Backend Inspector</span>
        </div>

        <div class="card">
            <div class="form-group">
                <label for="task">What would you like the agent to do?</label>
                <textarea id="task" placeholder="Example: Go to quotes.toscrape.com, find the first 3 quotes with author names, and summarize them."></textarea>
            </div>

            <div class="row">
                <div class="form-group">
                    <label for="model">LLM Backend</label>
                    <select id="model">
                        <option value="antigravity" selected>Antigravity (Auto CLI / API Key)</option>
                        <option value="browser-use">ChatBrowserUse</option>
                        <option value="google">Google Gemini</option>
                        <option value="openai">OpenAI GPT</option>
                    </select>
                </div>

                <div class="form-group">
                    <label>Browser Display</label>
                    <label class="checkbox-group">
                        <input type="checkbox" id="headless">
                        <span>Run Headless (hide Chrome window)</span>
                    </label>
                </div>
            </div>

            <div class="form-group">
                <label for="api-key">API Key <span style="font-weight: normal; color: var(--text-muted);">(Optional - paste Gemini / Browser Use / OpenAI key)</span></label>
                <input type="password" id="api-key" placeholder="AIzaSy... (leave blank to use CLI or .env file)">
            </div>

            <button id="run-btn" class="btn" onclick="startTask()">
                <div class="spinner" id="spinner"></div>
                <span id="btn-text">🚀 Run Browser Agent</span>
            </button>
        </div>

        <!-- Live Real-Time Activity & Timing Tracker -->
        <div class="live-panel" id="live-panel">
            <div class="live-header">
                <div class="status-tracker">
                    <div class="pulse-dot" id="pulse-dot"></div>
                    <span id="current-action-text">Initializing Agent...</span>
                </div>
                <div class="timer" id="timer-display">⏱️ 00:00</div>
            </div>
            <div class="terminal-box" id="terminal-box">
                <div class="log-entry">
                    <span class="log-time">[00:00]</span>
                    <span class="log-body">Starting backend execution stream...</span>
                </div>
            </div>
        </div>

        <!-- Execution Result Card -->
        <div class="card output-card" id="output-card">
            <div class="output-header">
                <span>EXECUTION RESULT</span>
                <span id="status-badge" class="status-badge badge-running">Running...</span>
            </div>
            <div class="output-content" id="output-text">Waiting for agent actions...</div>
        </div>
    </div>

    <script>
        let timerInterval = null;
        let startTime = 0;

        function startTimer() {
            startTime = Date.now();
            if (timerInterval) clearInterval(timerInterval);
            timerInterval = setInterval(() => {
                const elapsedSec = Math.floor((Date.now() - startTime) / 1000);
                const mins = String(Math.floor(elapsedSec / 60)).padStart(2, '0');
                const secs = String(elapsedSec % 60).padStart(2, '0');
                document.getElementById('timer-display').textContent = `⏱️ ${mins}:${secs}`;
            }, 500);
        }

        function stopTimer() {
            if (timerInterval) clearInterval(timerInterval);
        }

        function appendLog(timeStr, text, cssClass = '') {
            const box = document.getElementById('terminal-box');
            const entry = document.createElement('div');
            entry.className = 'log-entry';
            entry.innerHTML = `<span class="log-time">[${timeStr}]</span><span class="log-body ${cssClass}">${escapeHtml(text)}</span>`;
            box.appendChild(entry);
            box.scrollTop = box.scrollHeight;
        }

        function escapeHtml(str) {
            return String(str)
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;');
        }

        async function startTask() {
            const task = document.getElementById('task').value.trim();
            if (!task) {
                alert('Please enter a task description!');
                return;
            }

            const model = document.getElementById('model').value;
            const apiKey = document.getElementById('api-key').value.trim() || undefined;
            const headless = document.getElementById('headless').checked;

            const runBtn = document.getElementById('run-btn');
            const btnText = document.getElementById('btn-text');
            const spinner = document.getElementById('spinner');
            const livePanel = document.getElementById('live-panel');
            const currentActionText = document.getElementById('current-action-text');
            const pulseDot = document.getElementById('pulse-dot');
            const terminalBox = document.getElementById('terminal-box');
            const outputCard = document.getElementById('output-card');
            const outputText = document.getElementById('output-text');
            const statusBadge = document.getElementById('status-badge');

            runBtn.disabled = true;
            spinner.style.display = 'block';
            btnText.textContent = 'Agent Working...';
            livePanel.style.display = 'block';
            pulseDot.style.background = 'var(--primary)';
            pulseDot.style.boxShadow = '0 0 8px var(--primary)';
            currentActionText.textContent = 'Starting Chrome & Initializing LLM...';
            terminalBox.innerHTML = '';
            outputCard.style.display = 'none';

            startTimer();
            appendLog('00:00', 'Task submitted: ' + task, 'log-info');

            try {
                const response = await fetch('/api/run-stream', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ task, model, api_key: apiKey, headless, max_steps: 25 })
                });

                if (!response.ok) {
                    throw new Error('HTTP error ' + response.status + ': ' + response.statusText);
                }

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split('\\n');
                    buffer = lines.pop();

                    for (const line of lines) {
                        if (!line.startsWith('data: ')) continue;
                        const jsonStr = line.substring(6).trim();
                        if (!jsonStr) continue;

                        try {
                            const event = JSON.parse(jsonStr);
                            handleStreamEvent(event);
                        } catch (e) {
                            console.error('Parse error:', e, line);
                        }
                    }
                }
            } catch (err) {
                stopTimer();
                pulseDot.style.background = 'var(--error)';
                pulseDot.style.boxShadow = '0 0 8px var(--error)';
                currentActionText.textContent = 'Error occurred';
                appendLog('--:--', 'Connection error: ' + err.message, 'log-error');

                outputCard.style.display = 'block';
                statusBadge.className = 'status-badge badge-error';
                statusBadge.textContent = 'Failed';
                outputText.textContent = err.message;
            } finally {
                runBtn.disabled = false;
                spinner.style.display = 'none';
                btnText.textContent = '🚀 Run Browser Agent';
            }
        }

        function handleStreamEvent(event) {
            const currentActionText = document.getElementById('current-action-text');
            const pulseDot = document.getElementById('pulse-dot');
            const outputCard = document.getElementById('output-card');
            const outputText = document.getElementById('output-text');
            const statusBadge = document.getElementById('status-badge');

            const elapsedSec = Math.floor(event.elapsed || 0);
            const mins = String(Math.floor(elapsedSec / 60)).padStart(2, '0');
            const secs = String(elapsedSec % 60).padStart(2, '0');
            const timeTag = `${mins}:${secs}`;

            if (event.type === 'status') {
                currentActionText.textContent = event.text;
                appendLog(timeTag, event.text, 'log-info');
            } else if (event.type === 'step') {
                currentActionText.textContent = `Step ${event.step}: Deciding next action...`;
                appendLog(timeTag, `📍 Step ${event.step}: Analyzing page state`, 'log-step');
                if (event.thought) {
                    appendLog(timeTag, `💭 Thinking: ${event.thought}`, 'log-thought');
                }
                if (event.action) {
                    appendLog(timeTag, `⚡ Action: ${event.action}`, 'log-action');
                }
            } else if (event.type === 'log') {
                let css = '';
                const lower = event.text.toLowerCase();
                if (lower.includes('step')) css = 'log-step';
                else if (lower.includes('error') || lower.includes('failed') || lower.includes('exception')) css = 'log-error';
                else if (lower.includes('warn') || lower.includes('waiting')) css = 'log-warn';
                else if (lower.includes('navigate') || lower.includes('click') || lower.includes('action')) css = 'log-action';
                else if (lower.includes('querying') || lower.includes('antigravity') || lower.includes('llm')) css = 'log-info';

                const cleanText = event.text.replace(/\\x1b\\[[0-9;]*m/g, '');
                appendLog(timeTag, cleanText, css);

                if (cleanText.includes('📍 Step')) {
                    currentActionText.textContent = cleanText.substring(cleanText.indexOf('📍'));
                } else if (cleanText.includes('Querying Antigravity') || cleanText.includes('Waiting for LLM')) {
                    currentActionText.textContent = '🧠 LLM Thinking / Querying Backend...';
                } else if (cleanText.includes('Navigated to')) {
                    currentActionText.textContent = '🔗 Navigated to target page';
                }
            } else if (event.type === 'done') {
                stopTimer();
                outputCard.style.display = 'block';
                if (event.success) {
                    pulseDot.style.background = 'var(--success)';
                    pulseDot.style.boxShadow = '0 0 8px var(--success)';
                    currentActionText.textContent = `Completed in ${event.duration || elapsedSec}s!`;
                    statusBadge.className = 'status-badge badge-success';
                    statusBadge.textContent = 'Completed';
                    outputText.textContent = event.result || 'Task completed successfully.';
                    appendLog(timeTag, '✅ Task completed successfully!', 'log-step');
                } else {
                    pulseDot.style.background = 'var(--error)';
                    pulseDot.style.boxShadow = '0 0 8px var(--error)';
                    currentActionText.textContent = 'Execution Failed';
                    statusBadge.className = 'status-badge badge-error';
                    statusBadge.textContent = 'Error';
                    outputText.textContent = event.error || 'Task failed without explicit error.';
                    appendLog(timeTag, '❌ ' + (event.error || 'Execution failed'), 'log-error');
                }
            }
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def get_ui():
	return HTMLResponse(content=HTML_TEMPLATE)


@app.post("/api/run-stream")
async def run_task_stream(req: TaskRequest):
	"""Streaming endpoint providing real-time log, step, and timing telemetry via Server-Sent Events."""

	async def event_generator():
		start_time = time.time()
		queue: asyncio.Queue[dict] = asyncio.Queue()
		loop = asyncio.get_running_loop()

		log_handler = SSEQueueLogHandler(queue, loop, start_time)
		formatter = logging.Formatter('%(message)s')
		log_handler.setFormatter(formatter)
		root_logger = logging.getLogger()
		root_logger.addHandler(log_handler)

		yield f"data: {json.dumps({'type': 'status', 'text': 'Initializing agent & browser...', 'elapsed': 0.0})}\\n\\n"

		agent_task: asyncio.Task | None = None
		browser: Browser | None = None

		async def run_agent():
			nonlocal browser
			try:
				if req.model == 'antigravity':
					llm = ChatAntigravity(api_key=req.api_key)
				elif req.model == 'browser-use':
					llm = ChatBrowserUse(api_key=req.api_key) if req.api_key else ChatBrowserUse()
				elif req.model == 'google':
					llm = ChatGoogle(model='gemini-2.5-flash', api_key=req.api_key)
				elif req.model == 'openai':
					llm = ChatOpenAI(model='gpt-4.1-mini', api_key=req.api_key) if req.api_key else ChatOpenAI(model='gpt-4.1-mini')
				else:
					llm = ChatAntigravity(api_key=req.api_key)

				browser = Browser(headless=req.headless)

				async def on_new_step(browser_state: BrowserStateSummary, model_output, step_number: int):
					elapsed = time.time() - start_time
					thought = getattr(model_output, 'thinking', None) or getattr(model_output, 'next_goal', None) or ''
					actions = getattr(model_output, 'action', []) or []
					action_repr = ', '.join(str(a) for a in actions) if actions else 'Evaluating'
					queue.put_nowait({
						"type": "step",
						"step": step_number,
						"url": getattr(browser_state, 'url', ''),
						"thought": str(thought)[:200] if thought else '',
						"action": action_repr[:200],
						"elapsed": round(elapsed, 1),
					})

				agent = Agent(
					task=req.task,
					llm=llm,
					browser=browser,
					register_new_step_callback=on_new_step,
				)

				history = await agent.run(max_steps=req.max_steps)
				final_res = history.final_result()
				duration = round(time.time() - start_time, 1)

				if not history.is_successful():
					err_list = [e for e in history.errors() if e]
					if err_list:
						last_err = err_list[-1]
					else:
						last_action = history.last_action()
						last_err = getattr(last_action, 'error', None) if last_action else None
					queue.put_nowait({
						"type": "done",
						"success": False,
						"error": last_err or "Agent was unable to complete the task within max steps.",
						"duration": duration,
						"elapsed": duration,
					})
				else:
					queue.put_nowait({
						"type": "done",
						"success": True,
						"result": final_res or "Task completed successfully.",
						"duration": duration,
						"elapsed": duration,
					})
			except Exception as e:
				duration = round(time.time() - start_time, 1)
				queue.put_nowait({
					"type": "done",
					"success": False,
					"error": str(e),
					"duration": duration,
					"elapsed": duration,
				})

		agent_task = asyncio.create_task(run_agent())

		try:
			done = False
			while not done:
				try:
					item = await asyncio.wait_for(queue.get(), timeout=1.0)
					yield f"data: {json.dumps(item)}\\n\\n"
					if item.get("type") == "done":
						done = True
				except TimeoutError:
					elapsed = round(time.time() - start_time, 1)
					yield f"data: {json.dumps({'type': 'tick', 'elapsed': elapsed})}\\n\\n"

					if agent_task.done():
						while not queue.empty():
							rem = queue.get_nowait()
							yield f"data: {json.dumps(rem)}\\n\\n"
							if rem.get("type") == "done":
								done = True
						done = True
		finally:
			root_logger.removeHandler(log_handler)
			if agent_task and not agent_task.done():
				agent_task.cancel()

	return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/run")
async def run_task(req: TaskRequest):
	"""Fallback non-streaming execution endpoint."""
	try:
		if req.model == 'antigravity':
			llm = ChatAntigravity(api_key=req.api_key)
		elif req.model == 'browser-use':
			llm = ChatBrowserUse(api_key=req.api_key) if req.api_key else ChatBrowserUse()
		elif req.model == 'google':
			llm = ChatGoogle(model='gemini-2.5-flash', api_key=req.api_key)
		elif req.model == 'openai':
			llm = ChatOpenAI(model='gpt-4.1-mini', api_key=req.api_key) if req.api_key else ChatOpenAI(model='gpt-4.1-mini')
		else:
			llm = ChatAntigravity(api_key=req.api_key)

		browser = Browser(headless=req.headless)
		agent = Agent(
			task=req.task,
			llm=llm,
			browser=browser,
		)

		history = await agent.run(max_steps=req.max_steps)
		final_res = history.final_result()

		if not history.is_successful():
			err_list = [e for e in history.errors() if e]
			last_err = err_list[-1] if err_list else None
			return {
				"success": False,
				"error": last_err or "Agent was unable to complete the task within max steps."
			}

		return {"success": True, "result": final_res or "Task completed successfully."}
	except Exception as e:
		return {"success": False, "error": str(e)}


if __name__ == "__main__":
	port = int(os.environ.get("PORT", 8000))
	print("\n=======================================================")
	print(" Browser Use Web UI started successfully!")
	print(f" Open your browser at: http://localhost:{port}")
	print("=======================================================\n")
	uvicorn.run(app, host="127.0.0.1", port=port)
