import openai
import anthropic
import requests
import os
import json
from videochat import APIManager
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from dotenv import load_dotenv
from pathlib import Path

current_dir = Path(__file__).resolve().parent
dotenv_path = next(path for path in current_dir.parents if (path / '.env').exists())
load_dotenv(dotenv_path=dotenv_path / '.env')


openai.api_key = os.getenv["OPENAI_API_KEY"]
anthropic.api_key = os.getenv["CLAUDE_API_KEY"]
api_manager = APIManager()

# Most models return a response object: To recover the generated text, use 
#   response.choices[0].message.content
#
# But gpt-4-vision-preview returns a "requests" object: To recover the generated text, use
#   (response.json())['choices'][0]['message']['content']

default_response_object = {
    "id": "chatcmpl-abc123",
    "object": "chat.completion",
    "created": 1677858242,
    "model": "gpt-3.5-turbo-0613",
    "usage": {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0
    },
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": "Brain disconnect, sorry mate."
            },
            "logprobs": None,
            "finish_reason": "OpenAI API error",
            "index": 0
        }
    ]
}

class JSONToObject:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, JSONToObject(value) if isinstance(value, dict) else value)

def json_to_object(data):
    return json.loads(data, object_hook=JSONToObject)

# specific api calls will be moved to the api manager class
def get_api_response(messages):
    if api_manager.api == "anthropic":
        return api_manager.claude_api_call(messages, **kwargs)
    elif api_manager.api == "openai":
        return api_manager.openai_api_call(messages, **kwargs)
    elif api_manager.api == "ollama":
        return api_manager.ollama_api_call(messages, **kwargs)
    else:
        raise ValueError(f"Unsupported API: {api_manager.api}")

def get_response(messages, model="gpt-3.5-turbo", temperature=1.0, max_tokens=64, seed=1331, return_full_response=False):
    try:
        if model != "gpt-4-vision-preview":
            return get_api_response(messages, model, temperature, max_tokens, seed, return_full_response)
        else:
            return get_api_vision_response(messages, model, temperature, max_tokens, seed, return_full_response)
    except Exception: # all retries failed
        if(return_full_response):
            return default_response_object # todo: improve this!
        else:
            return "Brain disconnect, sorry mate."
        
headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {openai.api_key}"
}
endpoint_url = "https://api.openai.com/v1/chat/completions"

@retry(wait=wait_exponential(multiplier=1.5, min=1, max=60), stop=stop_after_attempt(6), retry=retry_if_exception_type(Exception))
def get_api_vision_response(messages, model="gpt-4-vision-preview", temperature=1.0, max_tokens=64, seed=1331, return_full_response=False):

    try:
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens
        }

        api_response = requests.post(endpoint_url, headers=headers, json=payload)
        json_response = api_response.json()

        if return_full_response:
            return json_response 
        else:
            return json_response['choices'][0]['message']['content'] # just the generated text
        
    except Exception as e:
        print(f"(API error: {e}, retrying...)")
        raise e
