ARG BUILD_FROM=ghcr.io/hassio-addons/base-python:3.12
FROM ${BUILD_FROM}

ENV PIP_DISABLE_PIP_VERSION_CHECK=1     PYTHONUNBUFFERED=1

RUN apk add --no-cache tzdata

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY app /app/app
COPY run.sh /run.sh
RUN chmod a+x /run.sh

CMD [ "/run.sh" ]
