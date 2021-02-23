FROM alekna/openbox-desktop-vnc AS GUI
RUN apt-get update; \
    apt-get install -y chromium; \
    apt-get clean
RUN mkdir /root/Desktop; \
    curl -sO https://download2.interactivebrokers.com/installers/tws/latest/tws-latest-linux-x64.sh; \
    echo "/root/Jts" | sh /tws-latest-linux-x64.sh; \
    rm /tws-latest-linux-x64.sh


ARG PYTHON_VERSION="3.7"
FROM python:$PYTHON_VERSION AS builder

#RUN pip3 install celery==4.4 flower requests psycopg2 ib_insync pandas numpy matplotlib
#WORKDIR /app
RUN pip3 install -U pip setuptools wheel


# By copying over requirements first, we make sure that Docker will cache
# our installed requirements rather than reinstall them on every build
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

RUN wget https://ixpeering.dl.sourceforge.net/project/ta-lib/ta-lib/0.4.0/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib \
    && ./configure --prefix=/usr \
    && make \
    && make install


#RUN pip install TA-Lib==0.4.19
RUN pip install TA-Lib


#WORKDIR .
FROM ubuntu:18.04 AS ubuntu18
MAINTAINER Joseph Loss <chicago-joe>


RUN  apt-get update \
  && apt-get install -y wget \
  && apt-get install -y unzip \
  && apt-get install -y xvfb \
  && apt-get install -y libxtst6 \
  && apt-get install -y libxrender1 \
  && apt-get install -y libxi6 \
  && apt-get install -y x11vnc \
  && apt-get install -y socat \
  && apt-get install -y software-properties-common \
  && apt-get install -y dos2unix

# Setup IB TWS
RUN mkdir -p /opt/TWS
WORKDIR /opt/TWS
RUN wget -q http://cdn.quantconnect.com/interactive/ibgateway-latest-standalone-linux-x64-v974.4g.sh
RUN chmod a+x ibgateway-latest-standalone-linux-x64-v974.4g.sh

# Setup  IBController
RUN mkdir -p /opt/IBController/ && mkdir -p /opt/IBController/Logs
WORKDIR /opt/IBController/
RUN wget -q http://cdn.quantconnect.com/interactive/IBController-QuantConnect-3.2.0.5.zip
RUN unzip ./IBController-QuantConnect-3.2.0.5.zip
RUN chmod -R u+x *.sh && chmod -R u+x Scripts/*.sh

WORKDIR /

# Install TWS
RUN yes n | /opt/TWS/ibgateway-latest-standalone-linux-x64-v974.4g.sh

ENV DISPLAY :0

ADD runscript.sh runscript.sh
ADD ./vnc/xvfb_init /etc/init.d/xvfb
ADD ./vnc/vnc_init /etc/init.d/vnc
ADD ./vnc/xvfb-daemon-run /usr/bin/xvfb-daemon-run

RUN chmod -R u+x runscript.sh \
  && chmod -R 777 /usr/bin/xvfb-daemon-run \
  && chmod 777 /etc/init.d/xvfb \
  && chmod 777 /etc/init.d/vnc

RUN dos2unix /usr/bin/xvfb-daemon-run \
  && dos2unix /etc/init.d/xvfb \
  && dos2unix /etc/init.d/vnc \
  && dos2unix runscript.sh

# Below files copied during build to enable operation without volume mount
COPY ../ML-IB-Trading-Docker/ib/IBController.ini /root/IBController/IBController.ini
COPY ../ML-IB-Trading-Docker/ib/jts.ini /root/Jts/jts.ini

CMD bash runscript.sh