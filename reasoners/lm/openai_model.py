import os
import openai
import numpy as np
from typing import Optional, Union, Literal
import time

from .. import LanguageModel, GenerateOutput
from openai import OpenAI, AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

PROMPT_TEMPLATE_ANSWER = "Your response need to be ended with \"So the answer is\"\n\n"
PROMPT_TEMPLATE_CONTINUE = "Please continue to answer the last question, following the format of previous examples. Don't say any other words.\n\n"

class OpenAIModel(LanguageModel):
    def __init__(self, model:str, max_tokens:int = 2048, temperature=0.0, additional_prompt=None, use_azure=False):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        if use_azure:
            token_provider = get_bearer_token_provider(
                DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
            )
            self.client = AzureOpenAI(
                azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", None),
                azure_ad_token_provider=token_provider,
                api_version = os.getenv("OPENAI_API_VERSION", None),
                api_key = os.getenv("AZURE_OPENAI_KEY", None)
            )
        else:
            self.client = OpenAI(
                api_key = os.getenv("OPENAI_API_KEY", None),
                # organization='',
            )
        self.additional_prompt = additional_prompt
    
    def generate(self,
                prompt: Optional[Union[str, list[str]]],
                max_tokens: int = None,
                top_p: float = 1.0,
                num_return_sequences: int = 1,
                rate_limit_per_min: Optional[int] = 20,
                stop: Optional[str] = None,
                logprobs: Optional[int] = None,
                temperature = None,
                additional_prompt=None,
                retry = 64,
                system_prompt=None,
                **kwargs) -> GenerateOutput:
        
        gpt_temperature = self.temperature if temperature is None else temperature
        if isinstance(prompt, list):
            assert len(prompt) == 1
            prompt = prompt[0]
        if additional_prompt is None and self.additional_prompt is not None:
            additional_prompt = self.additional_prompt
        elif additional_prompt is not None and self.additional_prompt is not None:
            print("Warning: additional_prompt set in constructor is overridden.")
        if additional_prompt == "ANSWER":
            prompt = PROMPT_TEMPLATE_ANSWER + prompt
        elif additional_prompt == "CONTINUE":
            prompt = PROMPT_TEMPLATE_CONTINUE + prompt

        if max_tokens is None:
            max_tokens = self.max_tokens
        
        if logprobs is None:
            logprobs = False
        else:
            logprobs = True


        for i in range(1, retry + 1):
            try:
                # sleep several seconds to avoid rate limit
                if rate_limit_per_min is not None:
                    time.sleep(60 / rate_limit_per_min)
                ### GPT 3.5 and higher use a different API
                if ('gpt-3.5' in self.model) or ('gpt-4' in self.model):
                    messages = [{"role": "user", "content": prompt}]
                    if system_prompt is not None:
                        messages = [{"role": "system", "content": system_prompt}] + messages
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        n=num_return_sequences,
                        stop=stop,
                        logprobs=logprobs
                    )
                    return GenerateOutput(
                        text=[choice.message.content for choice in response.choices],
                        log_prob=[choice.logprobs for choice in response.choices]
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        prompt=prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        n=num_return_sequences,
                        stop=stop,
                        logprobs=logprobs,
                        **kwargs
                    )
                    return GenerateOutput(
                        text=[choice["text"] for choice in response.choices],
                        log_prob=[choice["logprobs"] for choice in response["choices"]]
                    )
                # TODO: add support for logprobs
            
            except Exception as e:
                print(f"An Error Occured: {e}, sleeping for {i} seconds")
                time.sleep(i)
        
        # after 64 tries, still no luck
        raise RuntimeError("GPTCompletionModel failed to generate output, even after 64 tries")
    
    def get_next_token_logits(self,
                              prompt: Union[str, list[str]],
                              candidates: Union[list[str], list[list[str]]],
                              **kwargs) -> list[np.ndarray]:
        raise NotImplementedError("GPTCompletionModel does not support get_next_token_logits")

    def get_loglikelihood(self,
                          prompt: Union[str, list[str]],
                          **kwargs) -> list[np.ndarray]:
        raise NotImplementedError("GPTCompletionModel does not support get_log_prob")


if __name__ == '__main__':
    model = OpenAIModel(model='gpt-3.5-turbo')
    print(model.generate(['Hello, how are you?', 'How to go to Shanghai from Beijing?']))
