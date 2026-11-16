FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/ddsr-bench

RUN python -m pip install --no-cache-dir \
    "latex2sympy2-extended>=1.11,<2" \
    "numpy>=1.26,<3" \
    "regex>=2024.11,<2027" \
    "sympy>=1.13,<2"

COPY ddsr_bench /opt/ddsr-bench/ddsr_bench

WORKDIR /app
