# use nvidia triton server image
FROM nvcr.io/nvidia/tritonserver:23.08-py3

# Remove any third-party apt sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash

# Set working directory
WORKDIR /

# The base image comes with many system dependencies pre-installed to help you get started quickly.
# Please refer to the base image's Dockerfile for more information before adding additional dependencies.
# IMPORTANT: The base image overrides the default huggingface cache location.

# --- Optional: System dependencies ---
COPY builder/setup.sh /setup.sh
RUN /bin/bash /setup.sh && \
    rm /setup.sh

# Start Tritonserver
#COPY builder/run_tritonserver.sh /run_tritonserver.sh
#RUN /bin/bash /run_tritonserver.sh 

# Python dependencies
COPY builder/requirements.txt /requirements.txt
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --upgrade -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt

# NOTE: The base image comes with multiple Python versions pre-installed.
#       It is reccommended to specify the version of Python when running your code.

# Add src files (Worker Template)
ADD src .

CMD nohup tritonserver --model-repository /python_backend/models --http-port 3000 & python3 -u handler.py

CMD ["python3",  "-u", "handler.py"]