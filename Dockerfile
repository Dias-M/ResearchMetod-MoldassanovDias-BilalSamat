FROM python:3.11.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace/src \
    HF_HOME=/workspace/.cache/hf

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends make \
    && rm -rf /var/lib/apt/lists/*

# CPU build of torch first (the default PyPI wheel pulls multi-GB CUDA libraries).
# "torch==2.3.1" in requirements.txt is satisfied by "2.3.1+cpu".
RUN pip install --no-cache-dir torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default: run the offline benchmark on data/sample and write results/benchmark.log
CMD ["python", "src/benchmark.py", "--sample"]
