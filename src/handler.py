""" Example handler file with Triton. """

import runpod
import sys
import requests
import time


import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import *

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


def handler(job):
    """ Handler function that will be used to process jobs. """

    # wait until triton server returns ready
    url = "http://127.0.0.1:3000/v2/health/ready"  # Replace with your actual URL
    wait_for_ready(url)

    # run request
    job_input = job['input']

    STREAMING=False

    with grpcclient.InferenceServerClient("127.0.0.1:3000") as triton_client:

        try:
            infer_future = triton_client.async_infer(
                'tensorrt_llm',
                inputs,
                outputs=outputs,
                request_id=request_id,
                callback=partial(callback, user_data),
                parameters={'Streaming': False}
            )

            expected_responses = 1
            processed_count = 0

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
                    if FLAGS.return_log_probs:
                        cum_log_probs = result.as_numpy('cum_log_probs')
                        output_log_probs = result.as_numpy(
                            'output_log_probs')
                    if FLAGS.return_context_logits:
                        context_logits = result.as_numpy('context_logits')
                    if FLAGS.return_generation_logits:
                        generation_logits = result.as_numpy(
                            'generation_logits')
                    if output_ids is not None:
                        for beam_output_ids in output_ids[0]:
                            tokens = list(beam_output_ids)
                            actual_output_ids.append(tokens)
                    else:
                        print("Got cancellation response from server")

                processed_count = processed_count + 1





        except Exception as e:
            err = "Encountered error: " + str(e)
            print(err)
            sys.exit(err)

        passed = True

        for beam in range(FLAGS.beam_width):
            seq_len = sequence_lengths[0][beam] if (
                not FLAGS.streaming and len(sequence_lengths) > 0) else len(
                    actual_output_ids[beam])
            # These should be equal when input IDs are excluded from output
            output_ids_w_prompt = actual_output_ids[beam][:seq_len]
            output_ids_wo_prompt = (
                output_ids_w_prompt if FLAGS.exclude_input_in_output else
                output_ids_w_prompt[input_ids_data.shape[1]:])
            if tokenizer != None:
                output_text = tokenizer.decode(output_ids_wo_prompt)
                return {"response": output_text}

runpod.serverless.start({"handler": handler})
