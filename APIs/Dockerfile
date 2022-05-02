FROM python:3.7
WORKDIR /
RUN  apt update && apt upgrade  -y && apt install -y  netcat
COPY requirements.txt requirements.txt
RUN pip install -U pip wheel cmake
RUN pip install -r requirements.txt
COPY wait.sh  wait.sh
RUN ls
RUN chmod 777 wait.sh
RUN ls -alh
COPY . .
CMD ["sh","./wait.sh"]
