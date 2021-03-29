FROM nvidia/cuda:10.1-base
RUN apt-get update
RUN apt-get upgrade
RUN apt-get install wget