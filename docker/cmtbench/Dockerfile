FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/ddsr-bench

RUN python -m pip install --no-cache-dir \
    "numpy==2.2.5" \
    "regex==2024.11.6" \
    "sympy==1.13.3"

COPY ddsr_bench /opt/ddsr-bench/ddsr_bench

WORKDIR /app
