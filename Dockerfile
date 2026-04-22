FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md ./
COPY conf ./conf
COPY data ./data
COPY notebooks ./notebooks
COPY src ./src
COPY tests ./tests

RUN pip install --no-cache-dir -e .

CMD ["kedro", "run"]
