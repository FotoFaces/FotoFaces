import base64
import json
import sys
from random import choice
from argparse import ArgumentParser, FileType
from configparser import ConfigParser
from confluent_kafka import Producer, Consumer, OFFSET_BEGINNING
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg  

from flask import request, Flask
app = Flask(__name__)

# Topic for producing messages
TOPIC_PRODUCE = "image"
# Topic for consuming messages
TOPIC_CONSUME = "rev_image"

# Set up a callback to handle the '--reset' flag.
def reset_offset(consumer, partitions):
    if args.reset:
        for p in partitions:
            p.offset = OFFSET_BEGINNING
        consumer.assign(partitions)
        


@app.route("/", methods=["POST"])
def upload_image():
    # curl http://localhost:5000/ -d "candidate=<photo>" -d "id=<identifier>" -X POST
    
    # if there is a photo and a identifier
    if "candidate" in request.form.keys() and "id" in request.form.keys():
        img1 = request.form["candidate"]
        identifier = request.form['id']
        
        # GET photo from the database
        # produce a json message to send to the consumer
        producer.produce(TOPIC_PRODUCE, json.dumps({"command": "get_photo", "id": identifier}))
        producer.flush()
    
        # Poll for new messages from Kafka and save the json object
        msg_json = None
        try:
            while True:
                msg = consumer.poll(1.0)
                if msg is None:
                    pass
                elif msg.error():
                    print(f"ERROR Recieving GET from the Database: {msg.error()}")
                else:
                    msg_json = json.loads(msg.value().decode('utf-8'))
                    print(f"Consumed event from topic {TOPIC_CONSUME}: message = {msg_json}")
                    break
        except KeyboardInterrupt:
            return False
        
        # idk why it needs this but it doesn't work without it
        msg_json = json.loads(msg_json)
        # old photo from the database
        old_photo = msg_json["photo"]
        
        # decode_cropped = base64.b64decode(old_photo)
        # img = mpimg.imread(decode_cropped)
        # imgplot = plt.imshow(img)
        # plt.show()
        
        
        #
        # do fotofaces algorithms
        #
        
   
    return ("", 204)



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

    consumer.subscribe([TOPIC_CONSUME], on_assign=reset_offset)
    
    app.run(port=5002)