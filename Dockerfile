FROM python:3.10.4-slim

RUN apt-get update && \
    apt-get install -y \
        build-essential \
        python3-dev \
        python3-setuptools \
        ffmpeg \
        libsm6 \
        libxext6 \
        poppler-utils \
        make \
        gcc

COPY ./requirements.txt /requirements.txt

RUN python3 -m pip install -r requirements.txt

RUN apt-get remove -y --purge make gcc build-essential \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# COPY ./entrypoint.sh /entrypoint.sh
# RUN chmod +x entrypoint.sh

RUN mkdir src
COPY ./.TOKEN src/.TOKEN
COPY .models src/.models
COPY utils src/utils
COPY main.py src/main.py

WORKDIR "src"
RUN pwd

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

