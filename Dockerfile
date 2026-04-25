FROM python:3.12-slim

WORKDIR /app

# Install system dependencies and uv
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && apt-get purge -y curl \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Add uv to PATH (this IS required for subsequent RUN commands)
ENV PATH="/root/.local/bin:$PATH"
ENV PATH="/app/.venv/bin:$PATH"


# Copy dependency files AND source code (needed for editable install with dynamic version)
COPY pyproject.toml uv.lock* ./
COPY src ./src
COPY conf ./conf
COPY entrypoints ./entrypoints
COPY data ./data

# Install dependencies
RUN uv sync --no-dev --frozen

# Expose UI port
EXPOSE 8050

CMD ["python", "entrypoints/training.py"]