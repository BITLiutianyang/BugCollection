import openai
import traceback

GPT_VERSION = "gpt-4o"
OPENAI_BASE = "your api url"
OPENAI_API_KEY = "your tokns"

CLIENT = openai.OpenAI(
    api_key= OPENAI_API_KEY,
    base_url=OPENAI_BASE,
)



def do_request(message):
    try:
        response = CLIENT.chat.completions.create(
            model=GPT_VERSION,
            messages=message,
            max_tokens=1024,
            temperature=0,
            top_p = 1,
            frequency_penalty = 0,
            presence_penalty = 0,
            logprobs = True
            
        )
        
    except Exception as e:
        traceback.print_exc()
        return None
    return response.choices[0].message.content
