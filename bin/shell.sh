#!/usr/bin/env bash
#
# Run docker container in bash shell session.

source bin/set_docker_runtime.sh

docker-compose run --entrypoint bash forecast-research
