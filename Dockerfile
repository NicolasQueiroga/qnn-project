FROM nvcr.io/nvidia/pytorch:23.10-py3
LABEL maintainer="https://github.com/NicolasQueiroga"

COPY ./requirements.txt /requirements.txt
COPY ./code /code

WORKDIR /code


RUN apt update && apt -y upgrade && \
    apt install -y python3.10-venv && \
    python -m venv /py && \
    /py/bin/pip install --upgrade pip && \
    /py/bin/pip install -r /requirements.txt