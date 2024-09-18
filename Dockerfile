FROM python:3.9.13

WORKDIR /

# TZ setup
ARG TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN mkdir -p /workdir

COPY lib/apt.txt .
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get -qq update && \
    xargs apt-get -qq install --no-install-recommends < apt.txt && \
    apt-get install tesseract-ocr libtesseract-dev libleptonica-dev pkg-config -y && \
    apt-get install tesseract-ocr -y && \
    apt-get install tesseract-ocr-rus -y && \
    apt-get install ffmpeg libsm6 libxext6  -y && \
    apt-get -qq clean && \
    apt-get autoremove -y --purge && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*

COPY lib/requirements.txt .
RUN pip install --progress-bar=off -U --no-cache-dir -r requirements.txt
RUN CPPFLAGS=-I/usr/local/include pip install tesserocr

RUN mkdir -p /ToRecognize
RUN mkdir -p /Recognized
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata

COPY lib /workdir
COPY lib/start_first_main.sh .

# ENTRYPOINT cd /workdir && /bin/bash