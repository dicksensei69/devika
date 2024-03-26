from openai import OpenAI as OAI

from src.config import Config

class OpenAI:
    def __init__(self):
        config = Config()
        api_key = config.get_openai_api_key()
        base_url = "http://127.0.0.1:5000/v1" #changed to point at local api
        self.client = OAI(
            api_key=api_key,
            base_url=base_url,
        )
        
    def inference(self, model_id: str, prompt: str) -> str:
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt.strip(),
                }
            ],
            model=model_id,
        )

        return chat_completion.choices[0].message.content
