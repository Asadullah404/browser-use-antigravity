# Browser Use with Antigravity & Web UI

Autonomous AI Agent for Web Automation powered by Google Antigravity and Chrome DevTools Protocol (CDP).

---

## 🌟 Highlights & Custom Features Added

- **Antigravity Dual-Backend Support (`ChatAntigravity`)**:
  - **API Mode**: Automatically connects using `ANTIGRAVITY_API_KEY`, `GEMINI_API_KEY`, or `GOOGLE_API_KEY` for high-throughput, low-latency execution.
  - **CLI Fallback Mode**: If no API key is set, automatically detects the installed Antigravity CLI (`agy` / `agy.exe`) and runs non-interactively without requiring an API key.
- **Interactive Web UI (`web_ui.py`)**:
  - Web control center running on `http://localhost:8000`.
  - Type tasks in plain English, select LLM backend, toggle headless mode, and view live results.
- **Interactive CLI (`examples/ui/command_line.py`)**:
  - Run tasks directly from your terminal with interactive prompts.

---

## 🚀 Quickstart

### 1. Requirements
- Python >= 3.11 (tested on Python 3.12)
- [`uv`](https://github.com/astral-sh/uv) package manager
- Google Chrome or Chromium

### 2. Environment Setup
```powershell
# Activate virtual environment
.venv\Scripts\activate

# Or install fresh using uv
uv venv --python 3.12
.venv\Scripts\activate
uv sync
```

### 3. Configure Environment Variables (Optional)
Copy `.env.example` to `.env`:
```powershell
Copy-Item .env.example .env
```
Inside `.env`, configure any of the following:
```bash
# Antigravity / Gemini Key (optional - CLI mode works without a key)
ANTIGRAVITY_API_KEY=your_key_here

# Or Browser Use Cloud / OpenAI / Anthropic
BROWSER_USE_API_KEY=your_browser_use_key_here
OPENAI_API_KEY=your_openai_key_here
```

---

## 🖥️ Usage Options

### Option 1: Web UI (No Coding Required)
Launch the local web server:
```powershell
uv run python web_ui.py
```
Open **`http://localhost:8000`** in your browser. Enter your task, toggle headless mode, and click **Run Browser Agent**.

### Option 2: Terminal Interactive Prompt
Run the CLI script:
```powershell
uv run python examples/ui/command_line.py
```
Or pass the query directly:
```powershell
uv run python examples/ui/command_line.py --query "Go to Hacker News and get the top story"
```

### Option 3: Python Script
```python
import asyncio
from browser_use import Agent, Browser, ChatAntigravity

async def main():
    agent = Agent(
        task="Go to https://news.ycombinator.com and extract the title of the #1 post.",
        llm=ChatAntigravity(),
        browser=Browser(headless=False),
    )
    result = await agent.run()
    print("Result:", result.final_result())

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 🧪 Testing

Run the test suite for the Antigravity backend:
```powershell
uv run pytest tests/ci/models/test_llm_antigravity.py
```

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

