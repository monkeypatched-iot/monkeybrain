#!/bin/bash
# wait-for-it.sh

host=$1
port=$2
shift 2

# Loop until the service is available
until nc -z "$host" "$port"; do
  echo "Waiting for $host:$port to be ready..."
  sleep 2
done

# Execute the command passed after the wait
exec "$@"
