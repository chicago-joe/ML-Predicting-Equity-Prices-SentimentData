FROM ubuntu:latest AS builder

RUN apt-get update
RUN apt-get install -y unzip dos2unix wget

# Setup IB TWS
WORKDIR /root
RUN wget -q http://cdn.quantconnect.com/interactive/ibgateway-latest-standalone-linux-x64-v974.4g.sh
RUN chmod a+x ibgateway-latest-standalone-linux-x64-v974.4g.sh

RUN wget -q --progress=bar:force:noscroll --show-progress https://github.com/IbcAlpha/IBC/releases/download/3.8.1/IBCLinux-3.8.1.zip -O ibc.zip
RUN unzip ibc.zip -d /opt/ibc
RUN chmod a+x /opt/ibc/*.sh /opt/ibc/*/*.sh




COPY run.sh run.sh
RUN dos2unix run.sh

# Application
FROM ubuntu:latest

RUN apt-get update
RUN apt-get install -y x11vnc xvfb socat

WORKDIR /root

COPY --from=builder /root/ibgateway-latest-standalone-linux-x64-v974.4g.sh ibgateway-latest-standalone-linux-x64-v974.4g.sh

# Install TWS
RUN yes n | ./ibgateway-latest-standalone-linux-x64-v974.4g.sh


#RUN yes n | ./install-ibgateway.sh

RUN mkdir .vnc
RUN x11vnc -storepasswd 1358 .vnc/passwd

COPY --from=builder /opt/ibc /opt/ibc
COPY --from=builder /root/run.sh run.sh

COPY ibc_config.ini ibc/config.ini

ENV DISPLAY :0
ENV TRADING_MODE paper
EXPOSE 4002
EXPOSE 5900

CMD ./run.sh

