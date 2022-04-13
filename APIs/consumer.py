import sys
from argparse import ArgumentParser, FileType
from configparser import ConfigParser
from confluent_kafka import Consumer, Producer, OFFSET_BEGINNING
import json
import sqlite3 as sql
import logging

# mock database created with sqlite3
with sql.connect('mydb') as con:
    cur = con.cursor()
    cur.execute("""CREATE TABLE if not exists User (
        id int primary key,
        email text,
        full_name text,
        password varchar,
        photo blob);""")
    cur.close()

# logger
logging.basicConfig(filename="api.log", level=logging.DEBUG)

def update_photo(identification, photo):

    # database call
    logging.debug("-- Begin -- Database call while updating old photo")
    try :
        with sql.connect('mydb') as con:
            cur = con.cursor()
            # UPDATE <Table of Users> SET <photo> = <inputed photo> WHERE <identification> = <inputed identification>
            cur.execute(f"UPDATE User SET photo = \'{photo}\' WHERE id = \'{identification}\'")
            con.commit()
            cur.close()
            logging.debug(f"Successful query")
    # in case we get an unexpected error
    except KeyError as k:
        logging.debug(f"Error: {k}")
        return json.dumps({"command": "upload_photo", "id": identification, "photo": photo, "error": k})

    logging.debug("-- End -- Database call while updating old photo")

    # return identification and photo
    return json.dumps({"command": "upload_photo", "status": "successful"})

def get_photo(identification):

    # database call
    logging.debug("-- Begin -- Database call while getting the old photo")
    try:
        with sql.connect('mydb') as con:
            cur = con.cursor()
            # SELECT * FROM <Table of Users> WHERE <identification> = <inputed identification>
            cur.execute(f"SELECT photo FROM User WHERE id = \'{identification}\'")
            result = cur.fetchall()
            cur.close()
            # in case we dont get any results
            if result == []:
                # debug
                logging.debug(f"{result}")
                return json.dumps({"command": "get_photo", "id": identification, "error": "result is empty"})
            logging.debug(f"Successful query")
    # in case we get an unexpected error
    except KeyError as k:
        logging.debug(f"Error: {k}")
        return json.dumps({"command": "get_photo", "id": identification, "error": k})

    logging.debug("-- End -- Database call while getting the old photo")

    # extract photo from the list of results
    # result len should be 1 and only 1
    photo = result[0][0]

    # return identification and photo
    return json.dumps({"command": "get_photo", "id": identification, "photo": photo})


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
    topic_consumer = "image"
    consumer.subscribe([topic_consumer], on_assign=reset_offset)
    
    
    
    # Produce topic.
    topic_producer = "rev_image"
    
    def delivery_callback(err, msg):
        if err:
            print('ERROR: Message failed delivery: {}'.format(err))
        else:
            print("Produced event to topic {topic}: key = {key:12}".format(
                topic=msg.topic(), key=msg.value().decode('utf-8')))



    # Poll for new messages from Kafka and print them.
    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                # Initial message consumption may take up to
                # `session.timeout.ms` for the consumer group to
                # rebalance and start consuming
                print("Waiting...")
            elif msg.error():
                print(f"ERROR: {msg.error()}")
            else:
                # Extract the (optional) key and value, and print.
                
                msg_json = json.loads(msg.value().decode('utf-8'))
                
                print("Consumed event from topic {topic}: message = {message:12}".format(
                    topic=msg.topic(), message=msg.value().decode('utf-8')))
                
                if msg_json["command"] == "update_photo":
                    new_msg = update_photo(int(msg_json["id"]), msg_json["photo"])
                elif msg_json["command"] == "get_photo":
                    new_msg = get_photo(int(msg_json["id"]))
                else:
                    new_msg = {"error": "command not recognized"}
                    
                # send new msg
                print(f"Send new message back -> msg: {new_msg}")
                producer.produce(topic_producer, json.dumps(new_msg), callback=delivery_callback)
                
                
    except KeyboardInterrupt:
        pass
    finally:
        # Leave group and commit final offsets
        consumer.close()