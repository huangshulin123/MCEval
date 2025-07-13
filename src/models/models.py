# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from abc import ABC
import torch
# from vllm import LLM, SamplingParams

class LMMBaseModel(ABC):
    """
    Abstract base class for language model interfaces.

    This class provides a common interface for various language models and includes methods for prediction.

    Parameters:
    -----------
    model : str
        The name of the language model.
    max_new_tokens : int
        The maximum number of new tokens to be generated.
    temperature : float
        The temperature for text generation (default is 0).
    device: str
        The device to use for inference (default is 'auto').

    Methods:
    --------
    predict(input_text, **kwargs)
        Generates a prediction based on the input text.
    __call__(input_text, **kwargs)
        Shortcut for predict method.
    """
    def __init__(self, model_name, max_new_tokens, temperature, device='auto'):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device

    def predict(self, input_text, **kwargs):
        if self.device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = self.device
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(device)

        outputs = self.model.generate(input_ids, 
                                     max_new_tokens=self.max_new_tokens, 
                                     temperature=self.temperature,
                                     do_sample=True,
                                     **kwargs)
        
        out = self.tokenizer.decode(outputs[0])
        return out

    def __call__(self, input_text, **kwargs):
        return self.predict(input_text, **kwargs)


class BaichuanModel(LMMBaseModel):
    """
    Language model class for the Baichuan model.

    Inherits from LMMBaseModel and sets up the Baichuan language model for use.

    Parameters:
    -----------
    model : str
        The name of the Baichuan model.
    max_new_tokens : int
        The maximum number of new tokens to be generated.
    temperature : float, optional
        The temperature for text generation (default is 0).
    device: str
        The device to use for inference (default is 'auto').

    Methods:
    --------
    predict(input_text, **kwargs)
        Generates a prediction based on the input text.
    """
    def __init__(self, model_name, max_new_tokens, temperature, device, dtype):
        super(BaichuanModel, self).__init__(model_name, max_new_tokens, temperature, device)
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, torch_dtype=dtype, device_map=device, use_fast=False, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=dtype, device_map=device, trust_remote_code=True)

class CultureBank(LMMBaseModel):
    def __init__(self, model_name, max_new_tokens, temperature, device, dtype):
        super(CultureBank, self).__init__(model_name, max_new_tokens, temperature, device)
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer, AutoModelForCausalLM
        path = self.model_name
        if self.model_name == "SALT-NLP/CultureBank-Llama2-SFT":
            path = "CultureBank-Llama2-7b-chat-merged"
        elif self.model_name == "CultureBank-Mixtral-SFT":
            path = "CultureBank-Mixtral-SFT"
        elif self.model_name == "CultureBank-Mixtral-DPO":
            path = "CultureBank-Mixtral-DPO"
        self.model = LLM(model=path, tensor_parallel_size=4, max_num_seqs=50, enforce_eager=True)
        self.tokenizer = AutoTokenizer.from_pretrained(path, padding_side='left')
        self.sampling_params = SamplingParams(temperature=0.7, max_tokens=15000)
    def predict(self, input_texts, **kwargs):
        device = "cuda"
        messages_batch = [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": input_text}
            ] for input_text in input_texts
        ]
        texts = [
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            ) for messages in messages_batch
        ]
        outputs = self.model.generate(prompts=texts, sampling_params=self.sampling_params)
        generated_ids = [output.outputs[0].token_ids for output in outputs]

        # 解码生成的输出
        responses = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return responses

class QwenModel(LMMBaseModel):
    """
    Language model class for the Baichuan model.

    Inherits from LMMBaseModel and sets up the Baichuan language model for use.

    Parameters:
    -----------
    model : str
        The name of the Baichuan model.
    max_new_tokens : int
        The maximum number of new tokens to be generated.
    temperature : float, optional
        The temperature for text generation (default is 0).
    device: str
        The device to use for inference (default is 'auto').

    Methods:
    --------
    predict(input_text, **kwargs)
        Generates a prediction based on the input text.
    """
    def __init__(self, model_name, max_new_tokens, temperature, device, dtype):
        super(QwenModel, self).__init__(model_name, max_new_tokens, temperature, device)
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from vllm import LLM, SamplingParams
        path = self.model_name
        self.model = LLM(model=path, tensor_parallel_size=4, max_num_seqs=50, enforce_eager=True)
        self.tokenizer = AutoTokenizer.from_pretrained(path, padding_side='left')
        self.sampling_params = SamplingParams(temperature=0.7, max_tokens=15000)
    def predict(self, input_texts, **kwargs):
        device = "cuda"
        messages_batch = [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": input_text}
            ] for input_text in input_texts
        ]

        # 应用聊天模板并生成文本
        texts = [
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            ) for messages in messages_batch
        ]
        outputs = self.model.generate(prompts=texts, sampling_params=self.sampling_params)
        generated_ids = [output.outputs[0].token_ids for output in outputs]

        # 解码生成的输出
        responses = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        return responses
class GemmaModel(LMMBaseModel):
    """
    Language model class for the Baichuan model.

    Inherits from LMMBaseModel and sets up the Baichuan language model for use.

    Parameters:
    -----------
    model : str
        The name of the Baichuan model.
    max_new_tokens : int
        The maximum number of new tokens to be generated.
    temperature : float, optional
        The temperature for text generation (default is 0).
    device: str
        The device to use for inference (default is 'auto').

    Methods:
    --------
    predict(input_text, **kwargs)
        Generates a prediction based on the input text.
    """
    def __init__(self, model_name, max_new_tokens, temperature, device, dtype):
        super(GemmaModel, self).__init__(model_name, max_new_tokens, temperature, device)
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,padding_side='left')

    def predict(self, input_texts, **kwargs):
        # messages = [
        #     {"role": "system",
        #      "content": "Please integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{}."},
        #     {"role": "user", "content": prompt}
        # ]
        # system_messages = {'role': "system", 'content': "You are a helpful assistant."}
        device = "cuda"
        messages_batch = [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": input_text}
            ] for input_text in input_texts
        ]
        model_inputs = self.tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True).to(device)

        # 生成输出
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=15000,
            temperature=0.7,
            do_sample=False
        )

        # 计算生成的输出部分
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        # 解码生成的输出
        responses = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        # print(f"{responses=}")

        return responses

class MixtralModel(LMMBaseModel):
    """
    Language model class for the Mixtral model.

    Inherits from LMMBaseModel and sets up the Mixtral language model for use.

    Parameters:
    -----------
    model : str
        The name of the Mixtral model.
    max_new_tokens : int
        The maximum number of new tokens to be generated.
    temperature : float
        The temperature for text generation (default is 0).
    device: str
        The device to use for inference (default is 'auto').
    dtype: str
        The dtype to use for inference (default is 'auto').
    """
    def __init__(self, model_name, max_new_tokens, temperature, device, dtype):
        super(MixtralModel, self).__init__(model_name, max_new_tokens, temperature, device)
        from transformers import AutoTokenizer, AutoModelForCausalLM
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     model_name,
        #     torch_dtype="auto",
        #     device_map="auto"
        # )
        from vllm import LLM, SamplingParams
        path = self.model_name
        self.model = LLM(model=path, tensor_parallel_size=4, max_num_seqs=50, enforce_eager=True)
        self.tokenizer = AutoTokenizer.from_pretrained(path, padding_side='left')
        self.sampling_params = SamplingParams(temperature=0.7, max_tokens=15000)
    def predict(self, input_texts, **kwargs):
        device = "cuda"
        messages_batch = [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": input_text}
            ] for input_text in input_texts
        ]

        # 应用聊天模板并生成文本
        texts = [
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            ) for messages in messages_batch
        ]
        outputs = self.model.generate(prompts=texts, sampling_params=self.sampling_params)
        generated_ids = [output.outputs[0].token_ids for output in outputs]

        # 解码生成的输出
        responses = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        return responses


class MistralModel(LMMBaseModel):
    """
    Language model class for the Mistral model.

    Inherits from LMMBaseModel and sets up the Mistral language model for use.

    Parameters:
    -----------
    model : str
        The name of the Mistral model.
    max_new_tokens : int
        The maximum number of new tokens to be generated.
    temperature : float
        The temperature for text generation (default is 0).
    device: str
        The device to use for inference (default is 'auto').
    dtype: str
        The dtype to use for inference (default is 'auto').
    """
    def __init__(self, model_name, max_new_tokens, temperature, device, dtype):
        super(MistralModel, self).__init__(model_name, max_new_tokens, temperature, device)
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, torch_dtype=dtype, device_map=device)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=dtype, device_map=device)

class LlamaModel(LMMBaseModel):
    """
    Language model class for the Llama model.

    Inherits from LMMBaseModel and sets up the Llama language model for use.

    Parameters:
    -----------
    model : str
        The name of the Llama model.
    max_new_tokens : int
        The maximum number of new tokens to be generated.
    temperature : float
        The temperature for text generation (default is 0).
    device: str
        The device to use for inference (default is 'auto').
    dtype: str
        The dtype to use for inference (default is 'auto').
    system_prompt : str
        The system prompt to be used (default is None).
    model_dir : str
        The directory containing the model files (default is None). If not provided, it will be downloaded from the HuggingFace model hub.
    """
    def __init__(self, model_name, max_new_tokens, temperature, device, dtype):
        super(LlamaModel, self).__init__(model_name, max_new_tokens, temperature, device)
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from vllm import LLM, SamplingParams
        path = self.model_name
        self.model = LLM(model=path, tensor_parallel_size=4, max_num_seqs=50, enforce_eager=True)
        self.tokenizer = AutoTokenizer.from_pretrained(path, padding_side='left')
        self.sampling_params = SamplingParams(temperature=0.7,max_tokens = 15000)
    def predict(self, input_texts, **kwargs):
        device = "cuda"
        messages_batch = [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": input_text}
            ] for input_text in input_texts
        ]

        # 应用聊天模板并生成文本
        texts = [
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            ) for messages in messages_batch
        ]
        # print(f"{texts=}")
        # 将文本批处理并移动到设备上
        outputs = self.model.generate(prompts=texts, sampling_params=self.sampling_params)
        generated_ids = [output.outputs[0].token_ids for output in outputs]

        # 解码生成的输出
        responses = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        return responses

class OpenAIModel(LMMBaseModel):
    """
    Language model class for interfacing with OpenAI's GPT models.

    Inherits from LMMBaseModel and sets up a model interface for OpenAI GPT models.

    Parameters:
    -----------
    model : str
        The name of the OpenAI model.
    max_new_tokens : int
        The maximum number of new tokens to be generated.
    temperature : float
        The temperature for text generation (default is 0).
    system_prompt : str
        The system prompt to be used (default is None).
    openai_key : str
        The OpenAI API key (default is None).

    Methods:
    --------
    predict(input_text)
        Predicts the output based on the given input text using the OpenAI model.
    """
    def __init__(self, model_name, max_new_tokens, temperature, system_prompt, openai_key):
        super(OpenAIModel, self).__init__(model_name, max_new_tokens, temperature)
        # self.openai_key = openai_key
        self.system_prompt = system_prompt
        if self.model_name == "deepseek-v3":
            self.openai_key = ""
        else:
            self.openai_key = ""

    def predict(self, input_text, **kwargs):
        
        from openai import OpenAI
        if self.model_name == "deepseek-v3":
            base_url = "https://api.deepseek.com/v1"
        else:
            base_url = "https://api.openai.com/v1"
        client = OpenAI(api_key=self.openai_key, base_url=base_url,timeout=300)
        
        if isinstance(input_text, list):
            messages = input_text
        elif isinstance(input_text, dict):
            messages = [input_text]
        else:
            messages = [{"role": "user", "content": input_text}]

        n = kwargs['n'] if 'n' in kwargs else 1
        temperature = kwargs['temperature'] if 'temperature' in kwargs else self.temperature
        max_new_tokens = kwargs['max_new_tokens'] if 'max_new_tokens' in kwargs else self.max_new_tokens

        retries = 0
        delay_seconds = 3
        while retries < 1:
            try:
                if self.model_name in ["deepseek-v3"]:
                    response = client.chat.completions.create(
                        model="deepseek-chat",
                        messages=messages,
                        temperature=0.7
                    )
                else:
                    response = client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=temperature,
                        max_completion_tokens=max_new_tokens,
                        n=n,
                    )
                solve = True
                break
            except Exception as e:
                print(e)
                from time import sleep
                print(f"Retrying in {delay_seconds} seconds...")
                sleep(delay_seconds)
                retries += 1
                print(f"{retries=}")
        if n > 1:
            result = [choice.message.content for choice in response.choices]
        else:
            result = response.choices[0].message.content
            if self.model_name in ["gpt-4o"]:
                return result
            return result, dict(response.usage)
                

class HuggingFaceModel(LMMBaseModel):
    """
    Language model class for interfacing with Hugging Face's models.

    Inherits from LMMBaseModel and sets up a model interface for Hugging Face models.

    Parameters:
    -----------
    model : str
        The name of the Hugging Face model.
    max_new_tokens : int
        The maximum number of new tokens to be generated.
    temperature : float
        The temperature for text generation (default is 0).
    device: str
        The device to use for inference (default is 'auto').
    dtype: str

    Parameters of predict method:
    ----------------
    input_text: str
        The input text.

    """
    def __init__(self, model_name, max_new_tokens, temperature, device, dtype):
        super().__init__(model_name, max_new_tokens, temperature, device)
        from transformers import AutoTokenizer, AutoModelForCausalLM
        model = model_name
        
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True, torch_dtype=dtype, device_map=device)
        self.model = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True, torch_dtype=dtype, device_map=device)

    
    def predict(self, input_text, **kwargs):
        if self.device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = self.device
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(device)

        outputs = self.model.generate(input_ids, 
                                     max_new_tokens=self.max_new_tokens, 
                                     temperature=self.temperature,
                                     **kwargs)
        
        out = self.tokenizer.decode(outputs[0])
        return out[len(input_text):]

class CustomAPIModel(LMMBaseModel):
    """
    Language model class for interfacing with custom API models.
    
    Inherits from LMMBaseModel and sets up a model interface for custom API models.
    
    """
    def __init__(self, model_name, max_new_tokens, temperature, api_key, *args, **kwargs):
        super().__init__(model_name, max_new_tokens, temperature, *args, **kwargs)
        self.api_key = api_key
    
    def predict(self, input_text, **kwargs):
        pass

class YourModel(ABC):
    """
    Language model class for interfacing with custom models.
    
    Inherits from LMMBaseModel and sets up a model interface for custom models.
    
    ----------
    Parameters:
    ckpt_path : str
        The path to the model checkpoint.
    max_new_tokens : int
        The maximum number of new tokens to be generated.
    temperature : float
        The temperature for text generation (default is 0).
    
    """
    
    def __init__(self, ckpt_path, max_new_tokens, temperature, *args, **kwargs):
        self.model = None
        self.tokenizer = None
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        
        self.load_model(ckpt_path)
    
    def load_model(self, ckpt_path):
        pass
    
    def predict(self, input_text, **kwargs):
        pass
    
