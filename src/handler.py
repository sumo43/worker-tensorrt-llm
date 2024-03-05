""" Example handler file with Triton. """

import runpod
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

import numpy as np
import tritonclient.grpc as grpcclient
from transformers import AutoTokenizer
from tritonclient.utils import InferenceServerException, np_to_triton_dtype

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

def prepare_inputs(input_ids_data, input_lengths_data, request_output_len_data,
                   beam_width_data, temperature_data, repetition_penalty_data,
                   presence_penalty_data, frequency_penalty_data,
                   streaming_data, end_id, pad_id, prompt_embedding_table_data,
                   prompt_vocab_size_data, lora_weights_data, lora_config_data,
                   return_log_probs_data, top_k_data, top_p_data,
                   draft_ids_data, return_context_logits_data,
                   return_generation_logits_data):
    inputs = [
        prepare_tensor("input_ids", input_ids_data),
        prepare_tensor("input_lengths", input_lengths_data),
        prepare_tensor("request_output_len", request_output_len_data),
        prepare_tensor("beam_width", beam_width_data),
        prepare_tensor("temperature", temperature_data),
        prepare_tensor("streaming", streaming_data),
        prepare_tensor("end_id", end_id),
        prepare_tensor("pad_id", pad_id),
        prepare_tensor("return_log_probs", return_log_probs_data),
        prepare_tensor("runtime_top_k", top_k_data),
        prepare_tensor("runtime_top_p", top_p_data),
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


def handler(job):
    """ Handler function that will be used to process jobs. """

    # wait until triton server returns ready
    url = "http://127.0.0.1:3000/v2/health/ready"  # Replace with your actual URL
    wait_for_ready(url)

    # run request
    job_input = job['input']

    STREAMING=False

    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m",
                                                  legacy=False,
                                                  padding_side='left',
                                                  trust_remote_code=True)
    text = "hello"
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

    input_ids = [tokenizer.encode(text)]

    input_ids_data = np.array(input_ids, dtype=np.int32)
    input_lengths = [[len(ii)] for ii in input_ids]
    input_lengths_data = np.array(input_lengths, dtype=np.int32)
    request_output_len = [[500]]
    request_output_len_data = np.array(request_output_len, dtype=np.int32)
    beam_width = [[1]]
    beam_width_data = np.array(beam_width, dtype=np.int32)
    top_k = [[1]]
    top_k_data = np.array(top_k, dtype=np.int32)
    top_p = [[1.]]
    top_p_data = np.array(top_p, dtype=np.float32)
    temperature = [[0.7]]
    temperature_data = np.array(temperature, dtype=np.float32)
    return_log_probs = [[False]]

    return_context_logits_data = None
    return_generation_logits_data = None
    repetition_penalty_data = None
    presence_penalty_data = None
    frequency_penalty_data = None
    streaming = [[False]]
    streaming_data = np.array(streaming, dtype=bool)

    draft_ids_data = None

    return_log_probs_data = np.array(return_log_probs, dtype=bool)


    inputs = prepare_inputs(
        input_ids_data, input_lengths_data, request_output_len_data,
        beam_width_data, temperature_data, repetition_penalty_data,
        presence_penalty_data, frequency_penalty_data, streaming_data,
        end_id_data, pad_id_data, prompt_embedding_table_data,
        prompt_vocab_size_data, lora_weights_data, lora_config_data,
        return_log_probs_data, top_k_data, top_p_data, draft_ids_data,
        return_context_logits_data, return_generation_logits_data)

    outputs = None
    request_id = ''

    user_data = UserData()

    def callback(user_data, result, error):
        if error:
            user_data._completed_requests.put(error)
        else:
            user_data._completed_requests.put(result)

    with grpcclient.InferenceServerClient("127.0.0.1:3001") as triton_client:

        cb = partial(callback, user_data)

        infer_future = triton_client.async_infer(
            'tensorrt_llm',
            inputs,
            outputs=outputs,
            request_id=request_id,
            callback=cb,
            parameters={'Streaming': False}
        )

        expected_responses = 1
        processed_count = 0
        actual_output_ids = []


        while processed_count < expected_responses:
            try:
                result = user_data._completed_requests.get()
                print("Got completed request", flush=True)
            except Exception:
                break

            if type(result) == InferenceServerException:
                if result.status() == "StatusCode.CANCELLED":
                    print("Request is cancelled")
                else:
                    print("Received an error from server:")
                    print(result)
                    raise result
            else:
                #check_output_names(FLAGS.requested_outputs, result)
                output_ids = result.as_numpy('output_ids')
                sequence_lengths = result.as_numpy('sequence_length')
                if output_ids is not None:
                    for beam_output_ids in output_ids[0]:
                        tokens = list(beam_output_ids)
                        actual_output_ids.append(tokens)
                else:
                    print("Got cancellation response from server")

            processed_count = processed_count + 1

        passed = True

        n_responses = 1

        batch = {
            "choices": [{"tokens": []} for _ in range(n_responses)],
        }


        for beam in range(1):
            seq_len = sequence_lengths[0][beam] if (
                not streaming and len(sequence_lengths) > 0) else len(
                    actual_output_ids[beam])
            # These should be equal when input IDs are excluded from output
            output_ids_w_prompt = actual_output_ids[beam][:seq_len]
            output_ids_wo_prompt = (
                output_ids_w_prompt[input_ids_data.shape[1]:])
            if tokenizer != None:
                output_text = tokenizer.decode(output_ids_wo_prompt)
                return {"response": output_text}

runpod.serverless.start({"handler": handler})
