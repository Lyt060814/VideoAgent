from openai import OpenAI
from environment.config.config import config
import httpx

def get_client(model_prefix=None, timeout=120):
    """Get OpenAI client with appropriate credentials based on model prefix"""
    # Get model-specific API key and base URL if available, otherwise use defaults
    if model_prefix and f"{model_prefix}_api_key" in config['llm']:
        api_key = config['llm'][f"{model_prefix}_api_key"]
        base_url = config['llm'][f"{model_prefix}_base_url"]
    else:
        api_key = config['llm']['api_key']
        base_url = config['llm']['base_url']

    # Create HTTP client with timeout and retry configuration
    http_client = httpx.Client(
        timeout=httpx.Timeout(timeout, connect=10.0, read=timeout, write=10.0),
        limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        transport=httpx.HTTPTransport(retries=3)
    )

    # Create and return client with timeout configuration
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=http_client,
        max_retries=3
    )

def deepseek(model="deepseek-v3", system=None, user=None, messages=None, timeout=120):
    # Get client for deepseek
    client = get_client("deepseek", timeout=timeout)

    if messages is not None:
        pass
    else:
        messages = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        if user is not None:
            messages.append({"role": "user", "content": user})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=1,
        response_format={"type": "json_object"},
        timeout=timeout
    )
    return response

def claude(model="claude-3-7-sonnet-20250219", system=None, user=None, messages=None, timeout=120):
    # Get client for claude
    client = get_client("claude", timeout=timeout)

    if messages is not None:
        pass
    else:
        messages = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        if user is not None:
            messages.append({"role": "user", "content": user})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=1,
        timeout=timeout
    )
    return response

def gemini(model="gemini-2.5-flash", system=None, user=None, messages=None, timeout=120):
    # Get client for gemini
    client = get_client("gemini", timeout=timeout)

    if messages is not None:
        pass
    else:
        messages = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        if user is not None:
            messages.append({"role": "user", "content": user})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=1,
        timeout=timeout
    )
    return response

def gpt(model="gpt-4o", system=None, user=None, messages=None, timeout=120):
    # Get client for gpt
    client = get_client("gpt", timeout=timeout)

    if messages is not None:
        pass
    else:
        messages = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        if user is not None:
            messages.append({"role": "user", "content": user})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=1,
        timeout=timeout
    )
    return response