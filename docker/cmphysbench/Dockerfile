FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/ddsr-bench

RUN python -m pip install --no-cache-dir \
    "latex2sympy2-extended>=1.11,<2" \
    "numpy>=1.26,<3" \
    "pint>=0.26,<1" \
    "scipy>=1.13,<2" \
    "sympy>=1.13,<2" \
    "timeout-decorator>=0.5,<1"

COPY ddsr_bench /opt/ddsr-bench/ddsr_bench

WORKDIR /app
