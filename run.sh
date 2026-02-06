#!/usr/bin/with-contenv bashio
set -e

export TZ="$(bashio::config 'timezone' 2>/dev/null || echo 'UTC')"

python -m app.main
