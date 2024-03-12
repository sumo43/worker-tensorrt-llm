""" Example handler file with Triton. """

import runpod
from uuid import uuid4
import sys
import requests
import time

import argparse
import csv
import os
import queue
import sys
import time
from functools import partial

from vllm.entrypoints.openai.protocol import ChatCompletionRequest, CompletionRequest

import numpy as np
#import tritonclient.grpc as grpcclient
import tritonclient.grpc.aio as grpcclient
from transformers import AutoTokenizer
from tritonclient.utils import InferenceServerException, np_to_triton_dtype
from constants import ChatCompletionResponseStreamChoice, ChatCompletionStreamResponse, DeltaMessage, UsageInfo
from utils import JobInput

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")

model_name = "pytorch"
shape = [1]

# If your handler runs inference on a model, load the model here.
# You will want models to be loaded into memory before starting serverless.

def wait_for_ready(url):
    while True:
        try:
            response = requests.get(url)
            response.raise_for_status()
            if response.status_code == 200:
                print(f"Received 200 OK response from {url}")
                break
        except requests.RequestException as e:
            print(f"Error: {e}")
        
        print(f"Waiting for {url} to return 200 OK...")
        time.sleep(0.1)  

class UserData:

    def __init__(self):
        self._completed_requests = queue.Queue()

def prepare_tensor(name, input):
    t = grpcclient.InferInput(name, input.shape,
                              np_to_triton_dtype(input.dtype))
    t.set_data_from_numpy(input)
    return t

def prepare_inputs(input_ids_data, request_output_len_data,
                   beam_width_data, temperature_data, repetition_penalty_data,
                   presence_penalty_data, frequency_penalty_data,
                   streaming_data, end_id, pad_id, prompt_embedding_table_data,
                   prompt_vocab_size_data, lora_weights_data, lora_config_data,
                   return_log_probs_data, top_k_data, top_p_data,
                   draft_ids_data, return_context_logits_data,
                   return_generation_logits_data):
    inputs = [
        prepare_tensor("text_input", input_ids_data),
        prepare_tensor("max_tokens", request_output_len_data),
        prepare_tensor("beam_width", beam_width_data),
        prepare_tensor("temperature", temperature_data),
        prepare_tensor("stream", streaming_data),
        prepare_tensor("end_id", end_id),
        prepare_tensor("pad_id", pad_id),
        prepare_tensor("return_log_probs", return_log_probs_data),
        prepare_tensor("top_k", top_k_data),
        prepare_tensor("top_p", top_p_data),
    ]
    if prompt_embedding_table_data is not None:
        inputs += [
            prepare_tensor("prompt_embedding_table",
                           prompt_embedding_table_data),
            prepare_tensor("prompt_vocab_size", prompt_vocab_size_data)
        ]
    if lora_weights_data is not None:
        inputs += [
            prepare_tensor("lora_weights", lora_weights_data),
            prepare_tensor("lora_config", lora_config_data),
        ]
    if repetition_penalty_data is not None:
        inputs += [
            prepare_tensor("repetition_penalty", repetition_penalty_data),
        ]
    if presence_penalty_data is not None:
        inputs += [
            prepare_tensor("presence_penalty", presence_penalty_data),
        ]
    if frequency_penalty_data is not None:
        inputs += [
            prepare_tensor("frequency_penalty", frequency_penalty_data),
        ]
    if draft_ids_data is not None:
        inputs += [
            prepare_tensor("draft_input_ids", draft_ids_data),
        ]
    if return_context_logits_data is not None:
        inputs += [
            prepare_tensor("return_context_logits",
                           return_context_logits_data),
        ]
    if return_generation_logits_data is not None:
        inputs += [
            prepare_tensor("return_generation_logits",
                           return_generation_logits_data),
        ]
    return inputs


async def async_stream_yield(model_name, inputs, outputs, request_id):
    yield {
        "model_name": model_name,
        "inputs": inputs,
        "outputs": outputs,
        "request_id": request_id
    }


async def chat_completion_stream_generator(model_name, inputs, outputs, request_id):
    response_role = "user"
    created_time = int(time.monotonic())
    chunk_object_type = "chat.completion.chunk"
    first_iteration = True

    # Send response for each token for each request.n (index)
    previous_texts = [""] *  1# request.n
    previous_num_tokens = [0] * 1 #* request.n
    finish_reason_sent = [False] * 1 #* request.n
    
    async with grpcclient.InferenceServerClient("127.0.0.1:3001") as triton_client:

        response_iterator = triton_client.stream_infer(
            inputs_iterator=async_stream_yield(model_name, inputs, outputs, request_id)
        )

        # for now its always 0 since we dont support batching yet
        i = 0

        first_iteration = True

        async for result, err in response_iterator: 

            if err:
                print(err)

            if first_iteration:
                role = "assistant" #self.get_chat_request_role(request)

                choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=DeltaMessage(role=role),
                            logprobs=None,
                            finish_reason=None)
                chunk = ChatCompletionStreamResponse(
                            id=request_id,
                            object=chunk_object_type,
                            created=created_time,
                            choices=[choice_data],
                            model=model_name)
                data = chunk.model_dump_json(exclude_unset=True)

                yield f"data: {data}\n\n"

                first_iteration = False

            # parse the output

            if result.get_output('text_output') is not None:
                output = result.as_numpy('text_output')
                delta_text = output[0].decode("utf8")
                logprobs = None

                # Send token-by-token response for each request.n
                choice_data = ChatCompletionResponseStreamChoice(
                    index=i,
                    delta=DeltaMessage(content=delta_text),
                    logprobs=logprobs,
                    finish_reason=None)
                chunk = ChatCompletionStreamResponse(
                    id=request_id,
                    object=chunk_object_type,
                    created=created_time,
                    choices=[choice_data],
                    model=model_name)
                data = chunk.model_dump_json(exclude_unset=True)
                yield f"data: {data}\n\n"

    # Send the finish response for each request.n only once
    # prompt_tokens = len(res.prompt_token_ids)

    prompt_tokens = 60
    completion_tokens = 10
    total_tokens = 70
    final_usage = UsageInfo(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens
    )
    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(content="fin"),
        logprobs=None,
        finish_reason="stop")
    chunk = ChatCompletionStreamResponse(
        id=request_id,
        object=chunk_object_type,
        created=created_time,
        choices=[choice_data],
        model=model_name)
    if final_usage is not None:
        chunk.usage = final_usage
    data = chunk.model_dump_json(exclude_unset=True,
                                 exclude_none=True)
    yield f"data: {data}\n\n"

    yield "data: [DONE]\n\n"


async def handle(job):
    """ Handler function that will be used to process jobs. """

    global tokenizer
    # wait until triton server returns ready
    url = "http://127.0.0.1:3000/v2/health/ready"  # Replace with your actual URL
    wait_for_ready(url)

    # run request
    job_input = job['input']

    ji = JobInput(job_input)

    request = ChatCompletionRequest(**ji.openai_input)

    prompt = tokenizer.apply_chat_template(
                conversation=request.messages,
                tokenize=False,
                add_generation_prompt=request.add_generation_prompt)

    print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n")

    print(request)

    print("\n\n\n\n\n\n")

    STREAMING=False

    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m",
                                                  legacy=False,
                                                  padding_side='left',
                                                  trust_remote_code=True)
    text = prompt
    tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.encode(tokenizer.pad_token,
                              add_special_tokens=False)[0]
    end_id = tokenizer.encode(tokenizer.eos_token,
                              add_special_tokens=False)[0]

    end_id_data = np.array([[0]], dtype=np.int32)
    pad_id_data = np.array([[0]], dtype=np.int32)

    prompt_embedding_table_data = None
    prompt_vocab_size_data = None
    lora_weights_data = None
    lora_config_data = None

    #input_ids = [tokenizer.encode(text)]
    #input_ids_data = #np.array(input_ids, dtype=np.int32)
    #input_ids_data = np.array_str(text)

    input_ids_data = [[prompt]]
    input_ids_data = np.array(input_ids_data).astype(object)

    #input_lengths = [[len(ii)] for ii in input_ids]
    #input_lengths_data = np.array(input_lengths, dtype=np.int32)
    request_output_len = [[request.max_tokens if request.max_tokens is not None else 0]] # actually max len
    request_output_len_data = np.array(request_output_len, dtype=np.int32)
    beam_width = [[1]]
    beam_width_data = np.array(beam_width, dtype=np.int32)
    top_k = [[request.top_k]]
    top_k_data = np.array(top_k, dtype=np.int32)
    top_p = [[request.top_p]]
    top_p_data = np.array(top_p, dtype=np.float32)
    temperature = [[request.temperature]]
    temperature_data = np.array(temperature, dtype=np.float32)
    return_log_probs = [[False]]

    return_context_logits_data = None
    return_generation_logits_data = None
    repetition_penalty_data = None
    presence_penalty_data = None
    frequency_penalty_data = None
    streaming = [[True]]
    streaming_data = np.array(streaming, dtype=bool)

    draft_ids_data = None

    return_log_probs_data = np.array(return_log_probs, dtype=bool)


    inputs = prepare_inputs(
        input_ids_data, request_output_len_data,
        beam_width_data, temperature_data, repetition_penalty_data,
        presence_penalty_data, frequency_penalty_data, streaming_data,
        end_id_data, pad_id_data, prompt_embedding_table_data,
        prompt_vocab_size_data, lora_weights_data, lora_config_data,
        return_log_probs_data, top_k_data, top_p_data, draft_ids_data,
        return_context_logits_data, return_generation_logits_data)

    outputs = None
    request_id = str(uuid4())

    return chat_completion_stream_generator("tensorrt_llm_bls", inputs, outputs, request_id)



async def _handle_chat_or_completion_request(job):
        if openai_request.openai_route == "/v1/chat/completions":
            request_class = ChatCompletionRequest
            generator_function = self.chat_engine.create_chat_completion
        elif openai_request.openai_route == "/v1/completions":
            request_class = CompletionRequest
            generator_function = self.completion_engine.create_completion
        
        try:
            request = request_class(
                **openai_request.openai_input
            )
        except Exception as e:
            yield create_error_response(str(e)).model_dump()
            return
        
        print("start gen")
        response_generator = await generator_function(request, DummyRequest())

        print("end gen")

        batch = []
        batch_token_counter = 0
        batch_size = BatchSize(self.default_batch_size, self.min_batch_size, self.batch_size_growth_factor)
    
        async for chunk_str in response_generator:
            if "data" in chunk_str:
                if self.raw_openai_output:
                    data = chunk_str
                elif "[DONE]" in chunk_str:
                    continue
                else:
                    data = json.loads(chunk_str.removeprefix("data: ").rstrip("\n\n")) if not self.raw_openai_output else chunk_str
                batch.append(data)
                batch_token_counter += 1
                if batch_token_counter >= batch_size.current_batch_size:
                    if self.raw_openai_output:
                        batch = "".join(batch)
                    yield batch
                    batch = []
                    batch_token_counter = 0
                    batch_size.update()
        if batch:
            if self.raw_openai_output:
                batch = "".join(batch)
            yield batch
        
