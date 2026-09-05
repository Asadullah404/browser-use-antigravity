"""Test Antigravity LLM backend."""

import pytest

from browser_use.llm.antigravity.chat import ChatAntigravity, _extract_json_string
from browser_use.llm.models import get_llm_by_name


def test_extract_json_string():
	"""Test JSON string extraction from markdown and raw text."""
	assert _extract_json_string('{"key": "value"}') == '{"key": "value"}'
	assert _extract_json_string('```json\n{"key": "value"}\n```') == '{"key": "value"}'
	assert _extract_json_string('Here is the output:\n```\n{"key": "value"}\n```\nDone.') == '{"key": "value"}'
	assert _extract_json_string('Some text before {"a": 1} some text after') == '{"a": 1}'


def test_antigravity_api_key_initialization():
	"""Test that providing an API key initializes API backend mode."""
	chat = ChatAntigravity(model='gemini-2.5-flash', api_key='test-api-key')
	assert chat.backend == 'api'
	assert chat.api_key == 'test-api-key'
	assert chat.provider == 'antigravity'
	assert chat.name == 'antigravity-gemini-2.5-flash'


def test_antigravity_cli_initialization_with_custom_path():
	"""Test that providing a cli_path initializes CLI backend mode."""
	import sys

	# Using current python executable as a valid executable file for path validation
	chat = ChatAntigravity(cli_path=sys.executable)
	assert chat.backend == 'cli'
	assert chat.cli_path == sys.executable
	assert chat.provider == 'antigravity'


def test_antigravity_neither_available(monkeypatch):
	"""Test that error is raised when neither API key nor CLI is available."""
	monkeypatch.delenv('ANTIGRAVITY_API_KEY', raising=False)
	monkeypatch.delenv('GEMINI_API_KEY', raising=False)
	monkeypatch.delenv('GOOGLE_API_KEY', raising=False)
	monkeypatch.delenv('ANTIGRAVITY_AGENTAPI_EXE', raising=False)
	monkeypatch.setattr('browser_use.llm.antigravity.chat.find_antigravity_cli', lambda: None)

	with pytest.raises(ValueError, match='No Antigravity API key found'):
		ChatAntigravity(api_key=None, cli_path=None)


def test_get_llm_by_name_antigravity():
	"""Test resolving antigravity via get_llm_by_name."""
	model = get_llm_by_name('antigravity')
	assert isinstance(model, ChatAntigravity)

	model_agy = get_llm_by_name('agy')
	assert isinstance(model_agy, ChatAntigravity)

