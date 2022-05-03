#!/bin/bash


while ! curl kafka:9092 ; do; sleep 7; done

python appFlask.py kafkaconfig.ini
