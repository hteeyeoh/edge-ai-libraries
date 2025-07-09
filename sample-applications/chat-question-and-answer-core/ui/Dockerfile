# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

FROM node:20-slim AS vite-app

RUN groupadd -g 101 user && \
    useradd -m -s /bin/bash -u 101 -g 101 user

USER user

WORKDIR /user/app

COPY package*.json /user/app/
COPY .npmrc /user/app/

RUN ["npm", "ci"]

COPY . .

RUN ["npm", "run", "build"]

FROM nginxinc/nginx-unprivileged:1.27.4


COPY --from=vite-app /user/app/dist /usr/share/nginx/html

COPY ./env.sh /docker-entrypoint.d/env.sh
# COPY ./nginx.conf /etc/nginx/conf.d/default.conf
COPY ./nginx.conf /etc/nginx/nginx.conf

USER root
RUN chmod +x /docker-entrypoint.d/env.sh

USER nginx
