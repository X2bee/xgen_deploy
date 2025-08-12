FROM vllm/vllm-openai:latest

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

COPY . /vllm
WORKDIR /vllm

RUN chmod +x /vllm/entrypoint.sh

ENTRYPOINT ["/bin/bash", "-c", "/vllm/entrypoint.sh; tail -f /dev/null"]
