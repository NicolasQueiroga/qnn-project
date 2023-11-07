#!/bin/bash

if [ "$1" = "qnn" ]; then
  HOST_PORT=8080
else
  echo "Invalid argument supplied"
  echo "Usage: ./exec_container.sh <container_name>"
  exit 1
fi

# Set the name of the container
CONTAINER_NAME=$1
IMAGE_NAME=$1

# echo a message that to exit the container and keep it running in the background DO NOT use CTRL+D, use CTRL+P+Q
echo "############################################################################################"
echo "To exit the container and keep it running in the background DO NOT use CTRL+D, use CTRL+P+Q"
echo "############################################################################################"

# Check if the container exists
if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
  # Check if the container is running
  if [ "$(docker inspect -f '{{.State.Running}}' $CONTAINER_NAME)" = "true" ]; then
    # Container is already running, attach to its terminal
    docker attach $CONTAINER_NAME
  else
    # Container is stopped, start it and attach to its terminal
    docker start $CONTAINER_NAME
    docker attach $CONTAINER_NAME
  fi
else
  # Container doesn't exist, create and start it, and attach to its terminal
  docker compose run -p $HOST_PORT:80 --name $CONTAINER_NAME $IMAGE_NAME
  docker attach $CONTAINER_NAME
fi