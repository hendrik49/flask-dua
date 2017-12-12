import os
import random
import time
from flask import Flask, request, render_template, session, flash, redirect, \
    url_for, jsonify
from flask.ext.mail import Mail, Message
from celery import Celery
import sys

import logging
LOG_FILENAME = 'example.log'
LEVELS = {'debug': logging.DEBUG,
          'info': logging.INFO,
          'warning': logging.WARNING,
          'error': logging.ERROR,
          'critical': logging.CRITICAL}
          
if len(sys.argv) > 1:
    level_name = sys.argv[1]
    level = LEVELS.get(level_name, logging.NOTSET)
    logging.basicConfig(filename = LOG_FILENAME, level = level)



app = Flask(__name__)
app.config['SECRET_KEY'] = 'top-secret!'

# Flask-Mail configuration
app.config['MAIL_SERVER'] = 'smtp.googlemail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = 'flask@example.com'

# Celery configuration
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'


# Initialize extensions
mail = Mail(app)

# Initialize Celery
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)


@celery.task
def send_async_email(msg):
    """Background task to send an email with Flask-Mail."""
    with app.app_context():
        mail.send(msg)


@celery.task(bind=True)
def long_task(self):
    """Background task that runs a long function with progress reports."""
    import arcpy
    import pandas as pd
    import numpy as np
    dataPath = "C:/Prog/banghendrik/Combinasi_654_Jabo_Lapan_modified.tif"
    modelPath = "C:/Prog/banghendrik/DataTest_decisionTree.pkl"
    outputPath = "C:/Prog/banghendrik/Combinasi_654_Jabo_Lapan_modified_clf.tif"
    rasterarray = arcpy.RasterToNumPyArray(dataPath)
    # proc = subprocess.Popen(
    #     ['dir'],             #call something with a lot of output so we can see it
    #     shell=True,
    #     stdout=subprocess.PIPE
    # )

    # for line in iter(proc.stdout.readline,''):
    #     yield line.rstrip() + '<br/>\n'

    data = np.array([rasterarray[0].ravel(), rasterarray[1].ravel(), rasterarray[2].ravel()])
    data = data.transpose()

    import pandas as pd
    print("Change to dataframe format")
    logging.info('Change to dataframe format')
    columns = ['band1','band2', 'band3']
    df = pd.DataFrame(data, columns=columns)

    print("Split data to 20 chunks")
    logging.info('Split data to 20 chunks')
    df_arr = np.array_split(df, 20)
    from sklearn.externals import joblib
    clf = joblib.load(modelPath) 
    kelasAll = []
    for i in range(len(df_arr)):
        
        print ("predicting data chunk-"+str(i))
        logging.info("predicting data chunk-"+str(i))
        kelas = clf.predict(df_arr[i])
        dat = pd.DataFrame()
        dat['kel'] = kelas
        print ("mapping to integer class")
        logging.info("mapping to integer class")
        mymap = {'awan':1, 'air':2, 'tanah':3, 'vegetasi':4}
        dat['kel'] = dat['kel'].map(mymap)

        band1Array = dat['kel'].values
        print ("extend to list")
        logging.info("extend to list")
        kelasAll.extend(band1Array.tolist())

    del df_arr
    del clf
    del kelas
    del dat
    del band1Array
    del data

    print ("change list to np array")
    logging.info("change list to np array")
    kelasAllArray = np.array(kelasAll, dtype=np.uint8)

    print ("reshaping np array")
    logging.info("reshaping np array")
    band1 = np.reshape(kelasAllArray, (-1, rasterarray[0][0].size))
    band1 = band1.astype(np.uint8)

    raster = arcpy.Raster(dataPath)
    inputRaster = dataPath

    spatialref = arcpy.Describe(inputRaster).spatialReference
    cellsize1  = raster.meanCellHeight
    cellsize2  = raster.meanCellWidth
    extent     = arcpy.Describe(inputRaster).Extent
    pnt        = arcpy.Point(extent.XMin,extent.YMin)

    del raster

    # save the raster
    print ("numpy array to raster ..")
    logging.info("numpy array to raster ..")
    out_ras = arcpy.NumPyArrayToRaster(band1, pnt, cellsize1, cellsize2)
    #out_ras.save(outputPath)
    #arcpy.CheckOutExtension("Spatial")
    print ("define projection ..")
    logging.info ("define projection ..")
    arcpy.CopyRaster_management(out_ras, outputPath)
    arcpy.DefineProjection_management(outputPath, spatialref)
    verb = ['Starting up', 'Booting', 'Repairing', 'Loading', 'Checking']
    adjective = ['master', 'radiant', 'silent', 'harmonic', 'fast']
    noun = ['solar array', 'particle reshaper', 'cosmic ray', 'orbiter', 'bit']
    message = ''
    total = random.randint(10, 50)
    for i in range(total):
        if not message or random.random() < 0.25:
            message = '{0} {1} {2}...'.format(random.choice(verb),
                                              random.choice(adjective),
                                              random.choice(noun))
        self.update_state(state='PROGRESS',
                          meta={'current': i, 'total': total,
                                'status': message})
        time.sleep(1)
    return {'current': 100, 'total': 100, 'status': 'Task completed!',
            'result': 42}


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html', email=session.get('email', ''))
    email = request.form['email']
    session['email'] = email

    # send the email
    msg = Message('Hello from Flask',
                  recipients=[request.form['email']])
    msg.body = 'This is a test email sent from a background Celery task.'
    if request.form['submit'] == 'Send':
        # send right away
        send_async_email.delay(msg)
        flash('Sending email to {0}'.format(email))
    else:
        # send in one minute
        send_async_email.apply_async(args=[msg], countdown=60)
        flash('An email will be sent to {0} in one minute'.format(email))

    return redirect(url_for('index'))


@app.route('/longtask', methods=['POST'])
def longtask():
    task = long_task.apply_async()
    return jsonify({}), 202, {'Location': url_for('taskstatus',
                                                  task_id=task.id)}


@app.route('/status/<task_id>')
def taskstatus(task_id):
    task = long_task.AsyncResult(task_id)
    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'current': 0,
            'total': 1,
            'status': 'Pending...'
        }
    elif task.state != 'FAILURE':
        response = {
            'state': task.state,
            'current': task.info.get('current', 0),
            'total': task.info.get('total', 1),
            'status': task.info.get('status', '')
        }
        if 'result' in task.info:
            response['result'] = task.info['result']
    else:
        # something went wrong in the background job
        response = {
            'state': task.state,
            'current': 1,
            'total': 1,
            'status': str(task.info),  # this is the exception raised
        }
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
