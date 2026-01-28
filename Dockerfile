FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code except for artifacts and logs
COPY . .

CMD ["experiments/run_hier_snmf_steering.sh", "--steps", "train", "--act-model-name", "gpt2-small"]