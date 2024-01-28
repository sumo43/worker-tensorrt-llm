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
    first = job_input.get('first', 1)
    second = job_input.get('second', 2)

    with httpclient.InferenceServerClient("127.0.0.1:3000") as client:
        input0_data = np.array(first).astype(np.float32).repeat(4)
        input1_data = np.array(second).astype(np.float32).repeat(4)
        inputs = [
            httpclient.InferInput(
                "INPUT0", input0_data.shape, np_to_triton_dtype(input0_data.dtype)
            ),
            httpclient.InferInput(
                "INPUT1", input1_data.shape, np_to_triton_dtype(input1_data.dtype)
            ),
        ]

        inputs[0].set_data_from_numpy(input0_data)
        inputs[1].set_data_from_numpy(input1_data)

        outputs = [
            httpclient.InferRequestedOutput("OUTPUT0"),
            httpclient.InferRequestedOutput("OUTPUT1"),
        ]

        response = client.infer(model_name, inputs, request_id=str(1), outputs=outputs)

        result = response.get_response()
        output0_data = response.as_numpy("OUTPUT0")
        output1_data = response.as_numpy("OUTPUT1")

        if not np.allclose(input0_data + input1_data, output0_data):
            print("pytorch example error: incorrect sum")
            sys.exit(1)

        if not np.allclose(input0_data - input1_data, output1_data):
            print("pytorch example error: incorrect difference")
            sys.exit(1)

        print("PASS: pytorch")

        return "INPUT0 ({}) + INPUT1 ({}) = OUTPUT0 ({})".format(
                input0_data, input1_data, output0_data)


runpod.serverless.start({"handler": handler})