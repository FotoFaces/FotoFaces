# FotoFaces
Facial Recognition API for the FotoFaces project

# Kafka
## Get Kafka
```
$ tar -xzf kafka_2.13-3.1.0.tgz
$ cd kafka_2.13-3.1.0
```

## Start the Kafka environment
Run the following commands in order to start all services in the correct order, in different terminals:

``` 
# Start the ZooKeeper service
# Note: Soon, ZooKeeper will no longer be required by Apache Kafka.
$ bin/zookeeper-server-start.sh config/zookeeper.properties

# Start the Kafka broker service
$ bin/kafka-server-start.sh config/server.properties

# Once all services above have successfully launched, you will have a basic Kafka environment running and ready to use. 
```

## Create a topic to store your events
Kafka is a distributed event streaming platform that lets you read, write, store, and process events (also called records or messages in the documentation) across many machines. These events are organized and stored in topics. Very simplified, a topic is similar to a folder in a filesystem, and the events are the files in that folder.

```
# Creating a topic
# Run the command in another terminal
$ bin/kafka-topics.sh --create --topic <name> --bootstrap-server <address:port>
```

```
# Run this command for the display of some information
$ bin/kafka-topics.sh --describe --topic <name> --bootstrap-server <address:port>
```

## Write some events into the topic
By default, each line you enter will result in a separate event being written to the topic. 

```
$ bin/kafka-console-producer.sh --topic <name> --bootstrap-server <address:port>
```

## Read the events
```
$ bin/kafka-console-consumer.sh --topic <name> --from-beginning --bootstrap-server <address:port>
```

## Terminate the Kafka environment
```
$ rm -rf /tmp/kafka-logs /tmp/zookeeper
``` 

## References
```
https://kafka.apache.org/quickstart
``` 