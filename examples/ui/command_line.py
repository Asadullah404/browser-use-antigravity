"""
To Use It:

Example 1: Using OpenAI (default), with default task: 'go to reddit and search for posts about browser-use'
python command_line.py

Example 2: Using OpenAI with a Custom Query
python command_line.py --query "go to google and search for browser-use"

Example 3: Using Anthropic's Claude Model with a Custom Query
python command_line.py --query "find latest Python tutorials on Medium" --provider anthropic

"""

import argparse
import asyncio
import os
import sys

# Ensure local repository (browser_use) is accessible
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from browser_use import Agent
from browser_use.browser import BrowserSession
from browser_use.tools.service import Tools


def get_llm(provider: str):
	if provider == 'antigravity':
		from browser_use import ChatAntigravity

		return ChatAntigravity()
	elif provider == 'browser-use':
		from browser_use import ChatBrowserUse

		return ChatBrowserUse()
	elif provider == 'anthropic':
		from browser_use.llm import ChatAnthropic

		api_key = os.getenv('ANTHROPIC_API_KEY')
		if not api_key:
			raise ValueError('Error: ANTHROPIC_API_KEY is not set. Please provide a valid API key.')

		return ChatAnthropic(model='claude-3-5-sonnet-20240620', temperature=0.0)
	elif provider == 'openai':
		from browser_use import ChatOpenAI

		api_key = os.getenv('OPENAI_API_KEY')
		if not api_key:
			raise ValueError('Error: OPENAI_API_KEY is not set. Please provide a valid API key.')

		return ChatOpenAI(model='gpt-4.1', temperature=0.0)

	else:
		raise ValueError(f'Unsupported provider: {provider}')


def parse_arguments():
	"""Parse command-line arguments."""
	parser = argparse.ArgumentParser(description='Automate browser tasks using an LLM agent.')
	parser.add_argument(
		'--query', type=str, help='The query to process', default=None
	)
	parser.add_argument(
		'--provider',
		type=str,
		choices=['antigravity', 'browser-use', 'openai', 'anthropic'],
		default='antigravity',
		help='The model provider to use (default: antigravity)',
	)
	return parser.parse_args()


def initialize_agent(query: str, provider: str):
	"""Initialize the browser agent with the given query and provider."""
	llm = get_llm(provider)
	tools = Tools()
	browser_session = BrowserSession()

	return Agent(
		task=query,
		llm=llm,
		tools=tools,
		browser_session=browser_session,
		use_vision=True,
		max_actions_per_step=1,
	), browser_session


async def main():
	"""Main async function to run the agent."""
	args = parse_arguments()
	query = args.query
	if not query:
		query = input('\n🤖 Enter the task for the browser agent: ').strip()
		if not query:
			print('No task provided. Exiting.')
			return

	agent, browser_session = initialize_agent(query, args.provider)

	await agent.run(max_steps=25)

	input('\nPress Enter to close the browser...')
	await browser_session.kill()


if __name__ == '__main__':
	asyncio.run(main())
