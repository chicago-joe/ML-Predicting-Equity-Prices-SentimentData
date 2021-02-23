
# --------------------------------------------------------------------------------------------------
# upload sql to postgres

# app.conf.update(C_FORCE_ROOT=True, CELERY_WORKER_SEND_TASK_EVENTS = True,
#                 # CELERY_ACCEPT_CONTENT = ['pickle', 'json', 'msgpack', 'yaml'],
#                 CELERY_SEND_EVENTS = True,
# CELERY_SEND_TASK_SENT_EVENT = True,
# # CELERY_IGNORE_RESULT = True,
# # CELERY_DEFAULT_EXCHANGE = 'default',
# CELERY_ACKS_LATE = True,
# CELERYD_PREFETCH_MULTIPLIER = 1,
# CELERY_CREATE_MISSING_QUEUES = True,)

def fnUploadSQL(df=None, conn=None, tblName=None, mode='REPLACE', colNames=None, unlinkFile=True):

    setLogging(LOG_FILE_NAME = 'upload %s-%s.txt' % (tblName, os.getpid()), level='INFO')

    curTime = dt.time(dt.now()).strftime("%H_%M_%S")
    tmpFile = setOutputFilePath(OUTPUT_SUBDIRECTORY = 'upload', OUTPUT_FILE_NAME = '%s %s-%s.txt' % (tblName, curTime, os.getpid()))
    c = conn.cursor()

    logging.info("Creating temp file: %s" % tmpFile)
    colsSQL = pd.read_sql('SELECT * FROM %s LIMIT 0;' % (tblName), conn).columns.tolist()

    if colNames:
        # check columns in db table vs dataframe
        colsDF = df[colNames].columns.tolist()
        colsDiff = set(colsSQL).symmetric_difference(set(colsDF))

        if len(colsDiff) > 0:
            logging.warning('----- COLUMN MISMATCH WHEN ATTEMPTING TO UPLOAD %s -----' % tblName)
            if len(set(colsDF) - set(colsSQL))> 0:
                logging.warning('Columns in dataframe not found in %s: \n%s' % (tblName, list((set(colsDF) - set(colsSQL)))))
            else:
                df[colsDF].to_csv(tmpFile, sep="\t", na_rep="\\N", float_format="%.8g", header=False, index=False, doublequote=False)
                try:
                    c.copy_from(open(tmpFile), tblName, columns = colsDF)
                    logging.debug('Data inserted successfully...')
                except (Exception, psycopg2.DatabaseError) as err:
                    # os.remove(tmp_df)
                    # pass exception to function
                    show_psycopg2_exception(err)
                    c.close()
                # query = """COPY '%s' FROM '%s' (DELIMITER('|')); '%s' %s INTO TABLE %s LINES TERMINATED BY '\r\n' (%s)""" % \
                #         (tmpFile.replace('\\','/'), mode, tblName, colsDF)
                # c.copy_from(tmpFile.replace'\\')

                # logging.debug(query)
                # rv = c.execute(query)
                logging.debug("Number of rows affected: %s" % len(df))
                return

    # check columns in db table vs dataframe
    colsDF = df.columns.tolist()
    colsDiff = set(colsSQL).symmetric_difference(set(colsDF))

    if len(colsDiff) > 0:
        logging.warning('----- COLUMN MISMATCH WHEN ATTEMPTING TO UPLOAD %s -----' % tblName)
        if len(set(colsSQL) - set(colsDF))> 0:
            logging.warning('Columns in %s not found in dataframe: %s' % (tblName, list((set(colsSQL) - set(colsDF)))))
        if len(set(colsDF) - set(colsSQL))> 0:
            logging.warning('Columns in dataframe not found in %s: %s' % (tblName, list((set(colsDF) - set(colsSQL)))))


    df[colsSQL].to_csv(tmpFile, sep="\t", na_rep="\\N", float_format="%.8g", header=False, index=False, doublequote=False)
    # f = open(tmpFile.replace('\\','/'), 'r')
    try:
        c.copy_from(open(tmpFile), tblName)
        logging.debug('Data inserted successfully...')
    except (Exception, psycopg2.DatabaseError) as err:
        # os.remove(tmp_df)
        # pass exception to function
        show_psycopg2_exception(err)
        c.close()


    # query = """
    #             COPY %s
    #             FROM '%s'
    #             (DELIMITER('|'))
    #             ;
    #             """ % (tblName, tmpFile.replace('\\', '/'), mode)

    # logging.debug(query)
    # rv = c.execute(query)
    logging.info("Number of rows affected: %s" % len(df))

    # if (unlinkFile.lower() == 'yes') | (unlinkFile.lower() == 'y'):
    if unlinkFile:
        os.unlink(tmpFile)
        logging.info("Deleting temporary file: {}".format(tmpFile))
    logging.info("DONE")
    return










# app.conf.update(
# CELERYD_PREFETCH_MULTIPLIER=1,
# CELERYD_CONCURRENCY=1,
# CELERY_ACKS_LATE=True,
# CELERY_TIMEZONE = 'US/Central'
# )

#
#
# @app.task
# def add(x, y):
#     return x + y
#
# @app.task
# def mul(x, y):
#     return x * y
#
# @app.task
# def xsum(numbers):
#     return sum(numbers)

# @app.on_after_configure.finalize
# def setup_periodic_tasks(sender, **kwargs):
#     # Calls test('hello') every 10 seconds.
#     sender.add_periodic_task(10.0, test.s('hello'), name='add every 10')
#
#     # Calls test('world') every 30 seconds
#     # sender.add_periodic_task(30.0, test.s('world'), expires=10)
#
#     # Executes every Monday morning at 7:30 a.m.
#     # sender.add_periodic_task(
#     #     crontab(hour=7, minute=30, day_of_week=1),
#     #     test.s('Happy Mondays!'),
#     # )

# @app.task
# def test(arg):
#     print(arg)
# #

# app.conf.update(
# CELERYD_PREFETCH_MULTIPLIER=1,
# CELERYD_CONCURRENCY=1,
# CELERY_ACKS_LATE=True,
# CELERY_TIMEZONE = 'US/Central'
# )

# @app.task
# def add(x, y):
#     return x + y
#
# @app.task
# def mul(x, y):
#     return x * y
#
# @app.task
# def xsum(numbers):
#     return sum(numbers)
#
# @app.on_after_configure.finalize
# def setup_periodic_tasks(sender, **kwargs):
#     # Calls test('hello') every 10 seconds.
#     sender.add_periodic_task(10.0, test.s('hello'), name='add every 10')
#
#     # Calls test('world') every 30 seconds
#     # sender.add_periodic_task(30.0, test.s('world'), expires=10)
#
#     # Executes every Monday morning at 7:30 a.m.
#     # sender.add_periodic_task(
#     #     crontab(hour=7, minute=30, day_of_week=1),
#     #     test.s('Happy Mondays!'),
#     # )
# #
# @app.task
# def test(arg):
#     print(arg)
# #


# app.conf.update( CELERY_WORKER_SEND_TASK_EVENTS = True,
#                 CELERY_ACCEPT_CONTENT = ['pickle', 'json', 'msgpack', 'yaml'],
                # CELERY_SEND_EVENTS = True,
# CELERY_SEND_TASK_SENT_EVENT = True,
# CELERY_IGNORE_RESULT = True,
# CELERY_DEFAULT_EXCHANGE = 'default',
# CELERY_ACKS_LATE = True,
# CELERYD_PREFETCH_MULTIPLIER = 1,
# CELERY_CREATE_MISSING_QUEUES = True,)