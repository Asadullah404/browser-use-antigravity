import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar, overload

from pydantic import BaseModel, ValidationError

from browser_use.llm.base import BaseChatModel
from browser_use.llm.exceptions import ModelProviderError
from browser_use.llm.google.chat import ChatGoogle
from browser_use.llm.messages import BaseMessage
from browser_use.llm.views import ChatInvokeCompletion

T = TypeVar('T', bound=BaseModel)
logger = logging.getLogger(__name__)

_JSON_BLOCK_RE = re.compile(r'```(?:json)?\s*([\s\S]*?)\s*```', re.IGNORECASE)


def find_antigravity_cli() -> str | None:
	"""Find the Antigravity CLI ('agy') binary path on the current system."""
	# 1. Explicit environment variable
	env_exe = os.environ.get('ANTIGRAVITY_AGENTAPI_EXE')
	if env_exe and os.path.isfile(env_exe):
		return env_exe

	# 2. System PATH lookup
	which_exe = shutil.which('agy')
	if which_exe:
		return which_exe

	which_agy_exe = shutil.which('agy.exe')
	if which_agy_exe:
		return which_agy_exe

	# 3. Known default locations
	candidates = [
		Path.home() / 'AppData' / 'Local' / 'agy' / 'bin' / 'agy.exe',
		Path.home() / 'AppData' / 'Local' / 'agy' / 'bin' / 'agy.EXE',
		Path.home() / 'AppData' / 'Local' / 'agy' / 'agy.exe',
		Path.home() / '.gemini' / 'bin' / 'agy.exe',
		Path.home() / '.gemini' / 'bin' / 'agy',
		Path('/usr/local/bin/agy'),
		Path('/usr/bin/agy'),
	]
	for candidate in candidates:
		if candidate.is_file():
			return str(candidate)

	return None


def _extract_json_string(text: str) -> str:
	"""Extract JSON string from text, stripping markdown code fences if present."""
	text = text.strip()
	match = _JSON_BLOCK_RE.search(text)
	if match:
		return match.group(1).strip()
	# If text starts with '{' or '[', attempt parsing directly
	start_idx = text.find('{')
	end_idx = text.rfind('}')
	if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
		return text[start_idx : end_idx + 1]


def _fit_cli_prompt(cli_path: str, base_prompt: str, instruction_suffix: str, max_cmd_len: int = 12000) -> list[str]:
	"""Ensure command line length does not exceed Windows CreateProcess limit (32,767 chars)."""
	if sys.platform != 'win32':
		return [cli_path, '--disable-slash-commands', '-p', base_prompt + instruction_suffix]

	# Calculate overhead of the command and instruction suffix
	overhead = len(subprocess.list2cmdline([cli_path, '--disable-slash-commands', '-p', instruction_suffix])) + 300
	available_chars = max(500, max_cmd_len - overhead)

	# Direct O(1) slice: keep first 30% and last 70% of available chars
	if len(base_prompt) > available_chars:
		half_left = int(available_chars * 0.30)
		half_right = int(available_chars * 0.70)
		base_prompt = (
			base_prompt[:half_left]
			+ '\n\n...[Page DOM content truncated for CLI command length limit]...\n\n'
			+ base_prompt[-half_right:]
		)

	cmd = [cli_path, '--disable-slash-commands', '-p', base_prompt + instruction_suffix]

	# Safety check: if quote escaping still pushed it slightly over, fast geometric reduction (at most 4-5 steps)
	cmd_len = len(subprocess.list2cmdline(cmd))
	while cmd_len > max_cmd_len and len(base_prompt) > 100:
		base_prompt = base_prompt[:max(100, int(len(base_prompt) * 0.70))]
		cmd = [cli_path, '--disable-slash-commands', '-p', base_prompt + instruction_suffix]
		cmd_len = len(subprocess.list2cmdline(cmd))

	if cmd_len > max_cmd_len:
		excess = cmd_len - max_cmd_len + 100
		instruction_suffix = instruction_suffix[:max(100, len(instruction_suffix) - excess)]
		cmd = [cli_path, '--disable-slash-commands', '-p', base_prompt + instruction_suffix]

	return cmd


@dataclass
class ChatAntigravity(BaseChatModel):
	"""
	Antigravity LLM backend for Browser Use.

	Priority:
	1. If an API key is present (via parameter, ANTIGRAVITY_API_KEY, GEMINI_API_KEY,
	   or GOOGLE_API_KEY), uses the direct API client (ChatGoogle).
	2. If no API key is provided, detects the Antigravity CLI ('agy') and uses it as
	   the execution backend.
	3. If neither is available, raises an informative error.
	"""

	model: str = 'gemini-2.5-flash'
	api_key: str | None = None
	cli_path: str | None = None
	temperature: float | None = None
	timeout: int = 120
	extra_kwargs: dict[str, Any] = field(default_factory=dict)

	_backend: str = field(init=False, default='unknown')
	_google_chat: ChatGoogle | None = field(init=False, default=None)

	def __post_init__(self):
		resolved_key = (
			self.api_key
			or os.environ.get('ANTIGRAVITY_API_KEY')
			or os.environ.get('GEMINI_API_KEY')
			or os.environ.get('GOOGLE_API_KEY')
		)

		if resolved_key:
			self._backend = 'api'
			self.api_key = resolved_key
			self._google_chat = ChatGoogle(
				model=self.model,
				api_key=self.api_key,
				temperature=self.temperature,
				**self.extra_kwargs,
			)
			logger.info(f'ChatAntigravity initialized using API Key mode with model {self.model}')
		else:
			resolved_cli = self.cli_path or find_antigravity_cli()
			if resolved_cli:
				self._backend = 'cli'
				self.cli_path = resolved_cli
				logger.info(f'ChatAntigravity initialized using CLI mode ({self.cli_path})')
			else:
				raise ValueError(
					'No Antigravity API key found (ANTIGRAVITY_API_KEY, GEMINI_API_KEY, GOOGLE_API_KEY) '
					"and Antigravity CLI ('agy') is not installed on this system. "
					'Please set an API key in your .env or install the Antigravity CLI.'
				)

	@property
	def provider(self) -> str:
		return 'antigravity'

	@property
	def name(self) -> str:
		return f'antigravity-{self.model}'

	@property
	def backend(self) -> str:
		return self._backend

	def _serialize_messages_for_cli(self, messages: list[BaseMessage]) -> str:
		"""Convert BaseMessage sequence into a consolidated prompt string for the CLI."""
		parts: list[str] = []
		for msg in messages:
			role = msg.__class__.__name__.replace('Message', '').upper()
			if hasattr(msg, 'text') and msg.text:
				parts.append(f'[{role}]:\n{msg.text}')
			elif hasattr(msg, 'content') and msg.content:
				parts.append(f'[{role}]:\n{msg.content}')
			else:
				parts.append(f'[{role}]:\n{msg}')
		return '\n\n'.join(parts)

	@overload
	async def ainvoke(
		self, messages: list[BaseMessage], output_format: None = None, **kwargs: Any
	) -> ChatInvokeCompletion[str]: ...

	@overload
	async def ainvoke(self, messages: list[BaseMessage], output_format: type[T], **kwargs: Any) -> ChatInvokeCompletion[T]: ...

	async def ainvoke(
		self, messages: list[BaseMessage], output_format: type[T] | None = None, **kwargs: Any
	) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]:
		t0 = time.time()
		if self._backend == 'api' and self._google_chat is not None:
			logger.info(f'🧠 [ChatAntigravity API] Querying Gemini {self.model}...')
			res = await self._google_chat.ainvoke(messages=messages, output_format=output_format, **kwargs)
			logger.info(f'⚡ [ChatAntigravity API] Gemini response received in {time.time() - t0:.1f}s')
			return res

		# CLI backend execution
		assert self.cli_path is not None, 'Antigravity CLI path is not set'

		base_prompt = self._serialize_messages_for_cli(messages)

		if output_format is None:
			instruction_suffix = ''
		else:
			schema_str = json.dumps(output_format.model_json_schema(), indent=2)
			instruction_suffix = (
				f'\n\nCRITICAL INSTRUCTION: You must respond ONLY with a single valid JSON object '
				f'strictly matching the following JSON Schema:\n{schema_str}\n'
				f'Do NOT include any explanations, markdown code blocks, or extra text before or after the JSON.'
			)

		# Try executing with fitted prompt, and if Windows WinError 206 still occurs, retry with smaller limits
		last_os_err = None
		proc = None
		for attempt_max_len in [12000, 8000, 4000]:
			cmd = _fit_cli_prompt(self.cli_path, base_prompt, instruction_suffix, max_cmd_len=attempt_max_len)
			cmd_line_len = len(subprocess.list2cmdline(cmd))
			try:
				proc = await asyncio.create_subprocess_exec(
					*cmd,
					stdin=asyncio.subprocess.DEVNULL,
					stdout=asyncio.subprocess.PIPE,
					stderr=asyncio.subprocess.PIPE,
				)
				logger.info(f'🧠 [ChatAntigravity CLI] Subprocess spawned (command length: {cmd_line_len} chars)...')
				break
			except OSError as e:
				last_os_err = e
				if getattr(e, 'winerror', None) == 206 or '206' in str(e) or 'filename or extension is too long' in str(e).lower():
					logger.warning(f'Windows command line length ({cmd_line_len}) exceeded limit {attempt_max_len}, reducing prompt size...')
					continue
				raise ModelProviderError(
					message=f'Failed to execute Antigravity CLI: {e}',
					model=self.name,
				) from e

		if proc is None:
			raise ModelProviderError(
				message=f'Failed to execute Antigravity CLI due to Windows command line limit: {last_os_err}',
				model=self.name,
			)

		logger.info(f'🧠 [ChatAntigravity CLI] Querying Antigravity CLI (prompt length: {len(base_prompt)} chars)...')

		# Heartbeat logger while waiting for CLI
		async def _heartbeat():
			elapsed = 0
			try:
				while True:
					await asyncio.sleep(5)
					elapsed += 5
					logger.info(f'⏳ [ChatAntigravity CLI] Waiting for LLM response ({elapsed}s elapsed)...')
			except asyncio.CancelledError:
				pass

		heartbeat_task = asyncio.create_task(_heartbeat())

		try:
			try:
				stdout_bytes, stderr_bytes = await asyncio.wait_for(
					proc.communicate(),
					timeout=float(self.timeout),
				)
			finally:
				heartbeat_task.cancel()

			logger.info(f'⚡ [ChatAntigravity CLI] LLM response received in {time.time() - t0:.1f}s')

			stdout = stdout_bytes.decode('utf-8', errors='replace').strip()
			stderr = stderr_bytes.decode('utf-8', errors='replace').strip()

			if proc.returncode != 0:
				err_combined = f'{stderr} {stdout}'.strip()
				if 'Eligibility check failed' in err_combined or 'Authentication required' in err_combined:
					auth_help = (
						'\n\n[Antigravity Auth Note]: The Antigravity CLI session needs re-authentication. '
						'Run `agy` in your terminal to log in, OR add `GEMINI_API_KEY=your_key` to your .env file '
						'to use the high-performance direct API mode.'
					)
					raise ModelProviderError(
						message=f'Antigravity CLI failed with code {proc.returncode}: {err_combined}{auth_help}',
						model=self.name,
					)

				raise ModelProviderError(
					message=f'Antigravity CLI failed with code {proc.returncode}: {stderr or stdout}',
					model=self.name,
				)

			if output_format is None:
				return ChatInvokeCompletion(completion=stdout, usage=None)

			raw_json = _extract_json_string(stdout)
			try:
				parsed = output_format.model_validate_json(raw_json)
			except ValidationError as e:
				raise ModelProviderError(
					message=f'Antigravity CLI returned invalid JSON for structured output schema: {e}\nRaw output: {stdout}',
					model=self.name,
				) from e

			return ChatInvokeCompletion(completion=parsed, usage=None)

		except TimeoutError as e:
			raise ModelProviderError(
				message=f'Antigravity CLI timed out after {self.timeout}s',
				model=self.name,
			) from e
		except ModelProviderError:
			raise
		except Exception as e:
			raise ModelProviderError(
				message=f'Failed to execute Antigravity CLI: {e}',
				model=self.name,
			) from e
