FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY requirements.txt .

# Install ps,top for monitoring and curl for using ntfy
RUN apt-get update && apt-get install -y \
    procps \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Mount the default location for pip cache
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Copy the rest of the application code
COPY . .

CMD ["experiments/run_hier_snmf_steering.sh", "--steps", "train", "--act-model-name", "gpt2-small"]
