# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# FROM intel/dlstreamer-pipeline-server:2025.2-ubuntu24-rc2
FROM intel/dlstreamer-pipeline-server:latest


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./src /home/pipeline-server/gvapython/publisher/

# ENV XDG_RUNTIME_DIR=/tmp/.xdg_runtime_dir


# EXPOSE 8080