ARG BUILD_FROM
FROM ${BUILD_FROM}

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

# python + pip + tzdata
RUN apk add --no-cache \
    tzdata \
    python3 \
    py3-pip

WORKDIR /app
COPY requirements.txt /app/
RUN python3 -m pip install --no-cache-dir -r requirements.txt

COPY app /app/app
COPY run.sh /run.sh
RUN chmod a+x /run.sh

CMD [ "/run.sh" ]
