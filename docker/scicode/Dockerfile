FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/ddsr-bench

RUN python -m pip install --no-cache-dir \
    "h5py>=3,<4" \
    "matplotlib>=3.9,<4" \
    "numpy>=1.26,<3" \
    "scipy>=1.13,<2" \
    "sympy>=1.13,<2"

COPY ddsr_bench /opt/ddsr-bench/ddsr_bench
COPY datasets/scicode/test_data.h5 /opt/scicode/test_data.h5

WORKDIR /app
