import json
import sys
from random import choice
from argparse import ArgumentParser, FileType
from configparser import ConfigParser
from confluent_kafka import Producer, Consumer, OFFSET_BEGINNING

if __name__ == '__main__':
    # Parse the command line.
    parser = ArgumentParser()
    parser.add_argument('config_file', type=FileType('r'))
    parser.add_argument('--reset', action='store_true')
    args = parser.parse_args()
    
    # Parse the configuration.
    # See https://github.com/edenhill/librdkafka/blob/master/CONFIGURATION.md
    config_parser = ConfigParser()
    config_parser.read_file(args.config_file)
    config = dict(config_parser['default'])
    
    # Create Producer instance
    producer = Producer(config)    

    # Create Consumer instance
    config.update(config_parser['consumer'])
    consumer = Consumer(config)
    
    # Set up a callback to handle the '--reset' flag.
    def reset_offset(consumer, partitions):
        if args.reset:
            for p in partitions:
                p.offset = OFFSET_BEGINNING
            consumer.assign(partitions)

    # Subscribe to topic
    topic_consumer = "rev_image"
    consumer.subscribe([topic_consumer], on_assign=reset_offset)
    

    # Optional per-message delivery callback (triggered by poll() or flush())
    # when a message has been successfully delivered or permanently
    # failed delivery (after retries).
    def delivery_callback(err, msg):
        if err:
            print('ERROR: Message failed delivery: {}'.format(err))
        else:
            print("Produced event to topic {topic}: key = {key:12}".format(
                topic=msg.topic(), key=msg.value().decode('utf-8')))

    # Produce data by selecting random values from these lists.
    topic = "image"
    
    producer.produce(topic, json.dumps({"command": "get_photo", "id": "1"}), callback=delivery_callback)
    producer.produce(topic, json.dumps({"command": "update_photo", "id": "1", "photo": "photo420"}), callback=delivery_callback)
    producer.produce(topic, json.dumps({"command": "get_photo", "id": "1"}), callback=delivery_callback)

    # Block until the messages are sent.
    producer.poll(10000)
    producer.flush()
    
    # Poll for new messages from Kafka and print them.
    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                print("Waiting...")
            elif msg.error():
                print(f"ERROR: {msg.error()}")
            else:
                msg_json = json.loads(msg.value().decode('utf-8'))
                print(f"Consumed event from topic {topic_consumer}: message = {msg_json}")

                # do stuff here
                break
              
    except KeyboardInterrupt:
        pass
    finally:
        # Leave group and commit final offsets
        consumer.close()