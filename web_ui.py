import os
from typing import Literal

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

load_dotenv()

from browser_use import Agent, Browser
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

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Browser Use - Task Control Center</title>
    <style>
        :root {
            --bg: #0f172a;
            --surface: #1e293b;
            --border: #334155;
            --primary: #3b82f6;
            --primary-hover: #2563eb;
            --text: #f8fafc;
            --text-muted: #94a3b8;
            --success: #22c55e;
            --error: #ef4444;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            padding: 40px 20px;
        }
        .container {
            width: 100%;
            max-width: 800px;
            display: flex;
            flex-direction: column;
            gap: 24px;
        }
        .header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid var(--border);
            padding-bottom: 16px;
        }
        .header h1 {
            font-size: 24px;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
        }
        .form-group {
            display: flex;
            flex-direction: column;
            gap: 8px;
            margin-bottom: 20px;
        }
        label {
            font-size: 14px;
            font-weight: 600;
            color: var(--text);
        }
        textarea, select, input {
            background: #090d16;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px 14px;
            color: var(--text);
            font-size: 15px;
            outline: none;
            transition: border-color 0.2s;
        }
        textarea:focus, select:focus {
            border-color: var(--primary);
        }
        textarea {
            resize: vertical;
            min-height: 120px;
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
            font-size: 16px;
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
        .output-card {
            display: none;
        }
        .output-header {
            font-size: 15px;
            font-weight: 600;
            color: var(--text-muted);
            margin-bottom: 12px;
            display: flex;
            justify-content: space-between;
        }
        .output-content {
            background: #090d16;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 16px;
            font-family: monospace;
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
            <span style="color: var(--text-muted); font-size: 14px;">Antigravity & Browser Automation</span>
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

        <div class="card output-card" id="output-card">
            <div class="output-header">
                <span>EXECUTION RESULT</span>
                <span id="status-badge" class="status-badge badge-running">Running...</span>
            </div>
            <div class="output-content" id="output-text">Waiting for agent actions...</div>
        </div>
    </div>

    <script>
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
            const outputCard = document.getElementById('output-card');
            const outputText = document.getElementById('output-text');
            const statusBadge = document.getElementById('status-badge');

            runBtn.disabled = true;
            spinner.style.display = 'block';
            btnText.textContent = 'Agent Working...';
            outputCard.style.display = 'block';
            statusBadge.className = 'status-badge badge-running';
            statusBadge.textContent = 'Running...';
            outputText.textContent = 'Starting Chrome browser and initializing agent...\\nThis may take a moment depending on the task.';

            try {
                const response = await fetch('/api/run', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ task, model, api_key: apiKey, headless, max_steps: 25 })
                });

                const data = await response.json();

                if (response.ok && data.success) {
                    statusBadge.className = 'status-badge badge-success';
                    statusBadge.textContent = 'Completed';
                    outputText.textContent = data.result || 'Task completed successfully with no textual output.';
                } else {
                    statusBadge.className = 'status-badge badge-error';
                    statusBadge.textContent = 'Error';
                    outputText.textContent = data.error || 'An unexpected error occurred.';
                }
            } catch (err) {
                statusBadge.className = 'status-badge badge-error';
                statusBadge.textContent = 'Network Error';
                outputText.textContent = 'Failed to communicate with local server: ' + err.message;
            } finally {
                runBtn.disabled = false;
                spinner.style.display = 'none';
                btnText.textContent = '🚀 Run Browser Agent';
            }
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def get_ui():
	return HTMLResponse(content=HTML_TEMPLATE)

@app.post("/api/run")
async def run_task(req: TaskRequest):
	try:
		# 1. Select LLM
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

		# 2. Browser instance
		browser = Browser(headless=req.headless)

		# 3. Agent
		agent = Agent(
			task=req.task,
			llm=llm,
			browser=browser,
		)

		history = await agent.run(max_steps=req.max_steps)
		final_res = history.final_result()

		if not history.is_successful():
			err_list = [e for e in history.errors() if e]
			if err_list:
				last_err = err_list[-1]
			else:
				last_action = history.last_action()
				last_err = getattr(last_action, 'error', None) if last_action else None
			return {
				"success": False,
				"error": last_err or "Agent was unable to complete the task within max steps. Check terminal logs for details."
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

