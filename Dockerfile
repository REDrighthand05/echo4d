FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.lock pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m pip install -r requirements.lock && python -m pip install --no-deps . \
    && useradd --uid 10001 --create-home echo4d
USER echo4d
WORKDIR /data
ENTRYPOINT ["echo4d"]
CMD ["--help"]
