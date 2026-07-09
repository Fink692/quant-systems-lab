FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY tests ./tests
COPY data ./data
COPY docs ./docs
COPY examples ./examples
COPY config ./config
COPY reports ./reports

RUN python -m pip install --upgrade pip \
    && python -m pip install -e ".[dev]"

CMD ["pytest"]
