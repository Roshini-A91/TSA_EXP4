# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 



### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:
#### Import necessary Modules and Functions
~~~
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
~~~

#### Load dataset

~~~
data=pd.read_csv('/content/AirPassengers.csv')
~~~

#### Declare required variables and set figure size, and visualise the data

~~~
N=1000
plt.rcParams['figure.figsize'] = [12, 6] #plt.rcParams is a dictionary-like object in Mat

X=data['#Passengers']
plt.plot(X)
plt.title('Original Data')
plt.show()
plt.subplot(2, 1, 1)
plot_acf(X, lags=len(X)/4, ax=plt.gca())
plt.title('Original Data ACF')
plt.subplot(2, 1, 2)
plot_pacf(X, lags=len(X)/4, ax=plt.gca())
plt.title('Original Data PACF')
plt.tight_layout()
plt.show()

~~~

#### Fitting the ARMA(1,1) model and deriving parameters

~~~
arma11_model = ARIMA(X, order=(1, 0, 1)).fit()
phi1_arma11 = arma11_model.params['ar.L1']
theta1_arma11 = arma11_model.params['ma.L1']

~~~

#### Simulate ARMA(1,1) Process

~~~
ar1 = np.array([1, -phi1_arma11])
ma1 = np.array([1, theta1_arma11])
ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)
plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Process')
plt.xlim([0, 500])
plt.show()

~~~

#### Plot ACF and PACF for ARMA(1,1)

~~~
plot_acf(ARMA_1)
plt.show()
plot_pacf(ARMA_1)
plt.show()

~~~

#### Fitting the ARMA(1,1) model and deriving parameters

~~~
arma22_model = ARIMA(X, order=(2, 0, 2)).fit()
phi1_arma22 = arma22_model.params['ar.L1']
phi2_arma22 = arma22_model.params['ar.L2']
theta1_arma22 = arma22_model.params['ma.L1']
theta2_arma22 = arma22_model.params['ma.L2']

~~~

#### Simulate ARMA(2,2) Process

~~~
ar2 = np.array([1, -phi1_arma22, -phi2_arma22])
ma2 = np.array([1, theta1_arma22, theta2_arma22])
ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=N*10)
plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2) Process')
plt.xlim([0, 500])
plt.show()

~~~

#### Plot ACF and PACF for ARMA(2,2)

~~~
plot_acf(ARMA_2)
plt.show()
plot_pacf(ARMA_2)
plt.show()
~~~
OUTPUT:
ORIGINAL DATA:

SIMULATED ARMA(1,1) PROCESS:

![Screenshot 2025-04-12 102707](https://github.com/user-attachments/assets/5deeb0a1-cea9-4da7-a20f-3c5f63112f82)

Partial Autocorrelation
![Screenshot 2025-04-12 102806](https://github.com/user-attachments/assets/0c53fcf1-0eff-4997-9367-69c12fb70544)


Autocorrelation
![Screenshot 2025-04-12 102748](https://github.com/user-attachments/assets/cd57d785-0978-431f-acd4-ffa23c71153f)


SIMULATED ARMA(1,1) PROCESS:
![Screenshot 2025-04-12 102817](https://github.com/user-attachments/assets/c579f161-2313-4add-8c8d-27c065e039ca)

Partial Autocorrelation
![Screenshot 2025-04-12 102847](https://github.com/user-attachments/assets/573ee1ee-86b3-4f88-85be-3e0d616f7be9)


Autocorrelation
![Screenshot 2025-04-12 102829](https://github.com/user-attachments/assets/9d1f3774-fae9-44d6-82e2-5f30a81574c7)


SIMULATED ARMA(2,2) PROCESS:
![Screenshot 2025-04-12 102911](https://github.com/user-attachments/assets/364c68ad-18f7-4e84-91a2-3f6891aa1385)

Partial Autocorrelation
![Screenshot 2025-04-12 102930](https://github.com/user-attachments/assets/d1892147-4aaf-49bc-b76e-7dbbef9c7fab)

Autocorrelation
![Screenshot 2025-04-12 102920](https://github.com/user-attachments/assets/82635c7c-4d72-46b0-a6c5-1a0d3428c510)




RESULT:
Thus, a python program is created to fir ARMA Model successfully.
