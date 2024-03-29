import os
import runpod
from utils import JobInput
from engine import OpenAITRTEngine


engine = OpenAITRTEngine()
async def handler(job):
    #job_input = JobInput(job["input"])
    #engine = OpenAIvLLMEngine if job_input.openai_route else vllm_engine
    results_generator = engine.generate(job)
    async for batch in results_generator:
        #print(f"BATCH: {batch}")
        yield batch

runpod.serverless.start(
    {
        "handler": handler,
        "concurrency_modifier": lambda x: 300,
        "return_aggregate_stream": True,
    }
)
