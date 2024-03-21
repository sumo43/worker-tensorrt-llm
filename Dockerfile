# use nvidia triton server image
FROM nvcr.io/nvidia/tritonserver:24.02-trtllm-python-py3

########## SETUP ##########

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash

########## /SETUP ##########

########## TRTLLM ##########

#RUN apt-get update && apt-get -y install python3.10 python3-pip openmpi-bin libopenmpi-dev git
#RUN pip3 install tensorrt_llm==0.8.0 -U --pre --extra-index-url https://pypi.nvidia.com
#RUN pip3 uninstall --yes mpmath && pip3 install mpmath==1.3.0

#RUN mkdir /TensorRT-LLM
#WORKDIR /TensorRT-LLM


RUN git clone https://github.com/NVIDIA/TensorRT-LLM.git -b rel /TensorRT-LLM
RUN git clone https://github.com/triton-inference-server/tensorrtllm_backend.git -b rel /tensorrtllm_backend
RUN pip3 install mpi4py huggingface_hub

########## /TRTLLM ##########

# Set working directory
WORKDIR /

COPY builder/build_llama_engine.sh /build_llama_engine.sh
RUN chmod +x /build_llama_engine.sh


# The base image comes with many system dependencies pre-installed to help you get started quickly.
# Please refer to the base image's Dockerfile for more information before adding additional dependencies.
# IMPORTANT: The base image overrides the default huggingface cache location.

# Start Tritonserver
#COPY builder/run_tritonserver.sh /run_tritonserver.sh
#RUN /bin/bash /run_tritonserver.sh 

#RUN apt-get update && apt-get -y install python3.10 python3-pip openmpi-bin libopenmpi-dev

# Python dependencies
COPY builder/requirements.txt /requirements.txt
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --upgrade -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt

# NOTE: The base image comes with multiple Python versions pre-installed.
#       It is reccommended to specify the version of Python when running your code.

# Add src files (Worker Template)
ADD src .
ADD triton_model_repo /triton_model_repo

RUN cp -r /triton_model_repo /tensorrtllm_backend/triton_model_repo

#CMD exec /bin/sh -c "trap : TERM INT; sleep 9999999999d & wait"

#CMD ./build_llama_engine.sh && ulimit -n 1000000 && nohup tritonserver --model-repository /triton_model_repo  && python3 -u handler.py 

RUN chmod +x /start.sh
CMD /start.sh
#ENTRYPOINT ["/bin/bash", "-c", "/build_mistonstonserver --model-repository /triton_model_repoerver --model-repository /triton_model_repotral_engine.sh"]
#/tensorrtllm_backend/scripts/launch_triton_server.py --world_size=1 --model_repo=/tensorrtllm_│azureuser@artem-build:~/worker-tensorrt-llm/builder$ ^C
#backend/triton_model_repo

#CMD nohup tritonserver --model-repository /triton_model_repo --http-port 3000 --grpc_port 3001 --metrics_port 3002 & python3 -u handler.py
#CMD ["python3",  "-u", "handler.py"]
