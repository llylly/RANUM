FROM ubuntu:18.04

MAINTAINER Linyi Li

ENV DOCKYARD_SRC=.

ENV DOCKYARD_SRVHOME=/srv

RUN apt-get update && apt-get -y upgrade
RUN apt-get install -y python3.6 python3-pip cmake
RUN pip3 install --upgrade pip

WORKDIR ${DOCKYARD_SRVHOME}
COPY ${DOCKYARD_SRC} ${DOCKYARD_SRVHOME}

RUN apt-get install -y locales
RUN touch /usr/share/locale/locale.alias
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    locale-gen
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

RUN pip install -r ${DOCKYARD_SRVHOME}/requirements.txt

EXPOSE 8000

WORKDIR $DOCKYARD_SRVHOME
