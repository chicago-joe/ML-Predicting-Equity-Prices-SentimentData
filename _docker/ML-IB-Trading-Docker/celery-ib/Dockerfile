ARG PYTHON_VERSION="3.7"
FROM python:$PYTHON_VERSION as builder

#RUN pip3 install celery==4.4 flower requests psycopg2 ib_insync pandas numpy matplotlib
RUN pip3 install -U pip setuptools wheel
RUN pip3 install billiard==3.6.1.0 celery==4.4.0 flower==0.9.3 kombu==4.6.7 redis==3.2.1 vine==1.3.0 pydevd-pycharm sklearn beautifulsoup4==4.9.3 DateTime==4.3 empyrical==0.5.5 fastai==2.1.7 h5py==3.1.0 html5lib==1.1 ib-insync==0.9.64 ipython==7.19.0 jsonschema==3.2.0 lxml==4.6.2 matplotlib==3.3.3 more-itertools==8.6.0 multitasking==0.0.9 mysql-connector-python==8.0.22 nest-asyncio==1.4.3 numexpr==2.7.1 numpy==1.19.3 pandas==1.1.4 pandas-datareader==0.9.0 pandas-market-calendars==1.6.1 pathlib==1.0.1 polygon-api-client==0.1.9 psycopg2==2.8.6 python-dateutil==2.8.1 pytz==2020.4 PyYAML==5.3.1 Quandl==3.5.3 requests==2.25.0 scikit-learn==0.23.2 scipy==1.5.4 seaborn==0.11.0 statsmodels==0.12.1 tables==3.6.1 tiingo==0.12.0 torch==1.7.0 torchaudio==0.7.0 torchvision==0.8.1 trading-calendars==2.0.0 urllib3==1.26.2 yfinance==0.1.55 zipp==3.4.0


#ENV TA_PREFIX="/opt/ta-lib-core"
#ENV TA_LIBRARY_PATH="$TA_PREFIX/lib" \
#    TA_INCLUDE_PATH="$TA_PREFIX/include"

RUN wget https://ixpeering.dl.sourceforge.net/project/ta-lib/ta-lib/0.4.0/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib \
    && ./configure --prefix=/usr \
    && make \
    && make install

#WORKDIR /src/ta-lib-core
#RUN apt-get update && apt-get install -y \
#        gfortran \
#        libfreetype6-dev \
#        libhdf5-dev \
#        liblapack-dev \
#        libopenblas-dev \
#        libpng-dev \
#    && rm -rf /var/lib/apt/lists/* \
#    && curl -fsSL http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
#    | tar xvz --strip-components 1 \
#    && ./configure --prefix="$TA_PREFIX" \
#    && make \
#    && make install
#
#WORKDIR /src/ta-lib-python
#COPY . .
#RUN python -m pip install -e . \
#    && python -c 'import numpy, talib; close = numpy.random.random(100); output = talib.SMA(close); print(output)' \
#    && python -m pip wheel --wheel-dir wheels .
#
#ARG RUN_TESTS="1"
#RUN if [ "$RUN_TESTS" -ne "0" ]; then \
#        python -m pip install -r requirements_test.txt \
#        && nosetests -w . ; \
#    else \
#        echo "Skipping tests\n" ; \
#    fi
#
# Build final image.
#FROM python:$PYTHON_VERSION-slim
#COPY --from=builder /src/ta-lib-python/wheels /opt/ta-lib-python/wheels
#COPY --from=builder /opt/ta-lib-core /opt/ta-lib-core
#RUN python -m pip install --no-cache-dir /opt/ta-lib-python/wheels/*.whl \
#    && python -c 'import numpy, talib; close = numpy.random.random(100); output = talib.SMA(close); print(output)'