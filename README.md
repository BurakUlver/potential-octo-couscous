# potential-octo-couscous

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model,svm
from sklearn.metrics import mean_squared_error, r2_score
import mysql.connector
import datetime


connection = mysql.connector.connect(host='192.168.1.168',
                                     port=3306,
                                     user='burak.ulver',
                                     password='******',
                                     database='new_schema')
metric_no =110


for i in range(0,3):
  
  for j in range(1,11):
    query = connection.cursor()

    x_view = 'SELECT MetricValueDate FROM new_schema.GraphView INNER JOIN new_schema.MetricDetail ON GraphView.MetricNo = MetricDetail.MetricNo AND GraphView.SubMetricNo = MetricDetail.SubMetricNo WHERE GraphView.idGraph =%s ORDER BY MetricValueDate;'%(metric_no)
    y_view = 'SELECT MetricValue FROM new_schema.GraphView INNER JOIN new_schema.MetricDetail ON GraphView.MetricNo = MetricDetail.MetricNo AND GraphView.SubMetricNo = MetricDetail.SubMetricNo WHERE GraphView.idGraph =%s ORDER BY MetricValueDate;'%(metric_no)
    query.execute(x_view)
    X = query.fetchall()

    query.execute(y_view)
    y = query.fetchall()

    numrows = query.rowcount

    # Load the diabetes dataset
    arr_X = np.array([datetime.timedelta(0, 1, 36000)])
    arr_X = np.zeros((numrows,1),dtype='datetime64[ns]')

    j = 0
    for i in range(0,numrows):
      arr_X[j] = X[i]
      j += 1
    

    arr_X1 = np.zeros((numrows,1),dtype=int)
    for i in range(0,numrows):
      arr_X1[i]=i
  

    # Split the data into training/testing sets
    arr_X1_train = np.array(arr_X1[:-5])
    arr_X1_test = np.array(arr_X1[-5:])

    # Split the targets into training/testing sets

    arr_y = np.zeros((numrows, 1),dtype='float64')
    j = 0
    for i in range(0, numrows):
      arr_y[j] = y[i]
      j += 1


    arr_y_train = np.array(arr_y[:-5])
    arr_y_test = np.array(arr_y[-5:])
    print(arr_y_test)


    # Create linear regression object
    regr = svm.LinearSVC()

    # Train the model using the training sets
    regr.fit(arr_X1_train, arr_y_train.ravel())

    # Make predictions using the testing set
    arr_y_pred = regr.predict(arr_X1_test)

    # The coefficients
    print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print("Mean squared error: %.2f"
          % mean_squared_error(arr_y_test, arr_y_pred))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(arr_y_test, arr_y_pred))

    # Plot outputs
    plt.figure(metric_no)

    plt.plot(arr_X1,y , label='Data', color='r', linewidth=7)
    plt.plot(arr_X1_test, arr_y_test, label='Test',  color='black' ,linewidth=2)
    plt.plot(arr_X1_test, arr_y_pred, label='Prediction', color='blue', linewidth=3)
    plt.xlabel('Hafta')
    
    plt.title(metric_no)

    plt.legend()
    plt.show()
    metric_no +=1

  metric_no = 211

connection.close()
