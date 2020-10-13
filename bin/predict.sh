#!/usr/bin/env bash
#
# Run prediction model.
#
# Run "bin/predict.sh --help" to see available options.

source bin/set_docker_runtime.sh

docker-compose run --rm --entrypoint python SERVICE_NAME \
  -m src.prediction.prediction_session "$@"
