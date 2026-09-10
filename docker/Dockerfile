FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/critpt-eval

RUN python -m pip install --no-cache-dir \
    "numpy>=1.26,<3" \
    "scipy>=1.13,<2" \
    "sympy>=1.13,<2"

COPY critpt_eval /opt/critpt-eval/critpt_eval

WORKDIR /app
