FROM nginx:alpine

MAINTAINER Kapil Viren Ahuja <k.v.ahuja@gmail.com>

# setup nginx (this is a container we should replace later)

ADD nginx.conf /etc/nginx/

RUN apk update \
    && apk upgrade \
    && apk add --no-cache bash \
    && adduser -D -H -u 1000 -s /bin/bash www-data \
    && rm /etc/nginx/conf.d/default.conf

CMD ["nginx"]

# mount application to /app

#-------------------
# install Python 3.5
# and dependencies

RUN apk add --no-cache \
        ca-certificates \
        python3 python3-dev \
        gcc \
        linux-headers build-base \
    && \
    python3 -m ensurepip && \
    pip3 install --upgrade pip setuptools

WORKDIR /app

COPY requirements.txt /
RUN pip install -r /requirements.txt

#-----------------
# install postgres

RUN echo "@edge http://nl.alpinelinux.org/alpine/edge/main" >> /etc/apk/repositories && \
    apk update && \
    apk add curl "libpq@edge<9.7" "postgresql-client@edge<9.7" "postgresql@edge<9.7" "postgresql-contrib@edge<9.7" && \
    mkdir /docker-entrypoint-initdb.d && \
    curl -o /usr/local/bin/gosu -sSL "https://github.com/tianon/gosu/releases/download/1.2/gosu-amd64" && \
    chmod +x /usr/local/bin/gosu && \
    apk del curl && \
    rm -rf /var/cache/apk/*

ENV LANG en_US.utf8
ENV PGDATA /var/lib/postgresql/data
VOLUME /var/lib/postgresql/data

COPY docker-entrypoint.sh /

#---------
# main app

# ports setup
EXPOSE 80 443 8080

ENV POSTGRES_USER vgstats
EXPOSE 5432 8080
WORKDIR /apps
# CMD ["sh", "-c", "pg_ctl start && python3 api/api.py"]
