import os
import logging
import json
import traceback


from dotenv import load_dotenv
from torch.cuda import device_count
from typing import AsyncGenerator

from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.protocol import ChatCompletionRequest, CompletionRequest, ErrorResponse

from utils import DummyRequest, JobInput, BatchSize, create_error_response
from constants import DEFAULT_MAX_CONCURRENCY, DEFAULT_BATCH_SIZE, DEFAULT_BATCH_SIZE_GROWTH_FACTOR, DEFAULT_MIN_BATCH_SIZE
from tokenizer import TokenizerWrapper
from config import EngineConfig

from generator import handle

class OpenAITRTEngine:
    def __init__(self):
        pass
        #self.served_model_name = os.getenv("OPENAI_SERVED_MODEL_NAME_OVERRIDE") or self.config["model"]
        #self.response_role = os.getenv("OPENAI_RESPONSE_ROLE") or "assistant"
        #self.tokenizer = vllm_engine.tokenizer
        #self.default_batch_size = vllm_engine.default_batch_size
        #self.batch_size_growth_factor, self.min_batch_size = vllm_engine.batch_size_growth_factor, vllm_engine.min_batch_size
        #self._initialize_engines()
        #self.raw_openai_output = bool(int(os.getenv("RAW_OPENAI_OUTPUT", 1)))

    async def generate(self, openai_request: JobInput):
    
        req = JobInput(openai_request["input"])
        print("request input")
        print(req)

        if req.openai_route == "/v1/models":
            yield await self._handle_model_request()
        elif req.openai_route in ["/v1/chat/completions", "/v1/completions"]:
            async for response in self._handle_chat_or_completion_request(openai_request):
                yield response
        else:
            yield create_error_response("Invalid route").model_dump()

    async def _handle_model_request(self):
        models = await self.chat_engine.show_available_models()
        return models.model_dump()

    async def _handle_chat_or_completion_request(self, openai_request: JobInput):
        req = JobInput(openai_request["input"])

        #if req.openai_route == "/v1/chat/completions":
        #    request_class = ChatCompletionRequest
        #    generator_function = self.chat_engine.create_chat_completion
        #elif req.openai_route == "/v1/completions":
        #    request_class = CompletionRequest
        #    generator_function = self.completion_engine.create_completion


        if req.openai_route == "/v1/chat/completions":
            request_class = ChatCompletionRequest
        elif req.openai_route == "/v1/completions":
            request_class = CompletionRequest

        #generator_function = handle

        request = request_class(
                **req.openai_input
            )

        #request = request_class()

        print(request)


        response_generator = await handle(openai_request)

        batch = []
        batch_token_counter = 0
        #batch_size = BatchSize(1, 1, 2) #self.default_batch_size, self.min_batch_size, self.batch_size_growth_factor)

        print("finish handle")
        print(response_generator)

        try:
            async for chunk_str in response_generator:
                print("inner loop")
                if "data" in chunk_str:
                    print("stuff")
                    #if self.raw_openai_output:
                    data = chunk_str
                    #elif "[DONE]" in chunk_str:
                    #    continue
                    #else:
                    #    data = chunk_str
                    #    #data = json.loads(chunk_str.removeprefix("data: ").rstrip("\n\n")) if not self.raw_openai_output else chunk_str
                    #batch.append(data)
                    #batch_token_counter += 1

                    yield data

                    # dont do the batch stuff for now
                    #if batch_token_counter >= batch_size.current_batch_size:
                    #    if self.raw_openai_output:
                    #        batch = "".join(batch)
                    #    yield batch
                    #    batch = []
                    #    batch_token_counter = 0
                    #    batch_size.update()
            if batch:
                #if self.raw_openai_output:
                batch = "".join(batch)
                yield batch
        except Exception as e:
            print(e)

            raise e
