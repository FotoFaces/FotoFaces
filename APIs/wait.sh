#!/bin/bash


while ! nc -vz kafka 9092
  do
    sleep 7
  done

python RESTApi.py
