ARG PYTHON_VERSION="3.7"
FROM python:$PYTHON_VERSION as builder

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
