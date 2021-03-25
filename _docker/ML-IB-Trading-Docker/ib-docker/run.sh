#!/bin/bash

set -e
set -o errexit

rm -f /tmp/.X0-lock

Xvfb :0 &
sleep 1

x11vnc -rfbport 5900 -display :0 -usepw -forever &
socat TCP-LISTEN:4003,fork TCP:localhost:4001,forever &

# Start this last and directly, so that if the gateway terminates for any reason, the container will stop as well.
# Retry behavior can be implemented by re-running the container.
/opt/ibc/scripts/ibcstart.sh $(ls ~/Jts/ibgateway) --gateway "--mode=paper" "--user=schran892" "--pw=Quant1984!"
