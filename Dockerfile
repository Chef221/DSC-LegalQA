# Reproducibility Docker image for UIT Data Science Challenge 2026 Task 2
# Base: Official PyTorch with CUDA 12.1 runtime (reproducibility choice; not claimed bit-identical to Kaggle host)
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Remove incompatible torchao if present in base image
RUN pip uninstall -y torchao 2>/dev/null || true

# Install locked production environment
COPY requirements-lock.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data for official scorer
RUN python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('punkt')"

# Copy source, configs, artifacts, and scripts
COPY configs/ ./configs/
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY artifacts/ ./artifacts/
COPY pyproject.toml .

# Install package in editable mode
RUN pip install --no-cache-dir -e .

# Run preflight verification (artifacts and strict environment lock)
RUN python scripts/verify_artifacts.py && python scripts/verify_environment.py --strict

# Create data and output mounts
RUN mkdir -p /app/data /app/output

CMD ["python", "scripts/run_inference.py", "--help"]
