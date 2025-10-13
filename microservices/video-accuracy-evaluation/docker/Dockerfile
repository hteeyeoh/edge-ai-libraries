# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

FROM python:3.11-slim AS python-base

ARG GID=1000
ARG UID=1000

# Setting env vars for python, pip and poetry
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VERSION=1.8.3 \
    # Poetry installation location
    POETRY_HOME="/opt/poetry" \
    # Create the virtual environment dir .venv in the project's root
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    # Disable interactive prompts
    POETRY_NO_INTERACTION=1 \
    # Location where packages gets installed
    SETUP_PATH="/opt/projsetup" \
    # Location where virtual env is created
    VENV_PATH="/opt/projsetup/.venv"

# Add poetry and venv to path
ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"

# Upgrade pip and setuptools to the latest version to fix vuln
RUN /usr/local/bin/pip install --upgrade pip setuptools

# Create a non-root user
RUN groupadd -g ${GID} appuser && \
    useradd -m -s /bin/bash -u ${UID} -g ${GID} appuser && \
    chown -R appuser:appuser $HOME


FROM python-base AS builder-base
RUN apt-get update && apt-get install --no-install-recommends -y curl build-essential

# Install poetry. Installs in $POETRY_HOME with version $POETRY_VERSION
RUN curl -sSL https://install.python-poetry.org | python -

WORKDIR $SETUP_PATH
COPY pyproject.toml poetry.lock ./
# Install runtime (prod) deps. Uses $POETRY_VIRTUALENVS_IN_PROJECT to create venv.
RUN poetry install --no-dev && \
    rm -rf ~/.cache/pypoetry/* ~/.cache/pip/*


# `dev` image is used during development / testing
FROM python-base AS dev
ENV FASTAPI_ENV=development

WORKDIR $SETUP_PATH

# Copy the poetry installation and the virtual env to current image
COPY --from=builder-base $POETRY_HOME $POETRY_HOME
COPY --from=builder-base $SETUP_PATH $SETUP_PATH

# Install dev dependencies. Prod dependencies are already installed.
RUN poetry install --with dev

# Source code will be mounted here
WORKDIR /my-app

USER appuser

EXPOSE 9000

CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "9000", "--reload"]


# `prod` image used for runtime
FROM python-base AS prod
ENV FASTAPI_ENV=production

WORKDIR /my-app

# Copy the poetry installation and the virtual env
COPY --from=builder-base $SETUP_PATH $SETUP_PATH

# Copy source code
COPY src ./src

# Set ownership to appuser
RUN chown -R appuser:appuser /my-app

USER appuser

EXPOSE 9000

CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "9000"]
