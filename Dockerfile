# Local builds can use a local copy of pennsieve-ai-utils instead of the
# version on PyPI. To do that, build with:
#   docker build --build-context utils=../pennsieve-ai-utils ...
# Without that option, this stage is empty and the PyPI version is used.
FROM scratch AS utils

FROM python:3.12-slim

WORKDIR /app

# Source must be in place before `pip install .` because pyproject.toml's
# [tool.setuptools.packages.find] points at src/ and the build fails without it.
COPY pyproject.toml ./
COPY src/ ./src/

RUN pip install --no-cache-dir .
COPY --from=utils / /tmp/pennsieve-ai-utils/
RUN if [ -f /tmp/pennsieve-ai-utils/pyproject.toml ]; then \
        pip install --no-cache-dir /tmp/pennsieve-ai-utils; \
    fi && rm -rf /tmp/pennsieve-ai-utils

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "-m", "scout.main"]
