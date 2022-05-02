#!/bin/bash


while ! curl kafka:9092 ; do; sleep 7; done

python API.py config_producer.ini
