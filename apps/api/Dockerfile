FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

ARG AIDLENS_EXTRAS=""

COPY pyproject.toml ./
COPY app ./app
RUN pip install --upgrade pip && \
    if [ -n "$AIDLENS_EXTRAS" ]; then \
      pip install ".[${AIDLENS_EXTRAS}]"; \
    else \
      pip install .; \
    fi

COPY alembic.ini ./
COPY alembic ./alembic

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
