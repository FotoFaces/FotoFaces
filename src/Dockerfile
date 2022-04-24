FROM python:3.7-slim
WORKDIR /
RUN  apt update && apt upgrade  -y && apt install -y gcc musl-dev cmake  librdkafka-dev libx11-dev python3-dev build-essential ffmpeg libsm6 libxext6 curl netcat
COPY requirements.txt requirements.txt
RUN pip install -U pip wheel cmake
RUN pip install -r requirements.txt
COPY wait.sh  wait.sh
RUN ls
RUN chmod 777 wait.sh
EXPOSE 5000
COPY . .
CMD ["./wait.sh"]
