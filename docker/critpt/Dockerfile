FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/ddsr-bench

RUN python -m pip install --no-cache-dir \
    "numpy>=1.26,<3" \
    "scipy>=1.13,<2" \
    "sympy>=1.13,<2" \
    "pydantic>=2.10,<3" \
    "httpx>=0.27,<1" \
    "pyyaml>=6,<7" \
    "jinja2>=3.1,<4"

COPY ddsr_bench /opt/ddsr-bench/ddsr_bench
COPY configs/__init__.py /opt/ddsr-bench/configs/__init__.py

WORKDIR /app
