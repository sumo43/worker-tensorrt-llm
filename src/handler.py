import os
import runpod
from utils import JobInput
from engine import vLLMEngine, OpenAITRTEngine


engine = OpenAITRTEngine
async def handler(job):

    #job_input = JobInput(job["input"])
    #engine = OpenAIvLLMEngine if job_input.openai_route else vllm_engine
    results_generator = engine.generate(job)
    async for batch in results_generator:
        yield batch

runpod.serverless.start(
    {
        "handler": handler,
        "concurrency_modifier": lambda x: 32,
        "return_aggregate_stream": True,
    }
)
