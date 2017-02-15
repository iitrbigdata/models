import model
import quandl
import pandas as pd
import numpy as np



#quandl dataset from database 
quandl.ApiConfig.api_key = "A_5rSzHkvw5r_SDmMhBr"
dataset = quandl.get("yahoo/googl", start_date="2005-12-31", end_date="2010-12-31")




data1 = pd.DataFrame(data= dataset,columns = ['Open','Volume','Close'])

data1['Date'] = data1.index
#taking only two colums
data2 = pd.DataFrame(data= data1,columns = ['Open','Volume'])
data2.index = np.arange(1,1260) 
data2 = data2/np.amax(data2)


#calling the class
NN = Ann(2,6)


y = pd.DataFrame(data= data1,columns = ['Close'])
y.index = np.arange(1,1260)
Y = y/np.amax(y)



#test and training the model with dataset from quandl
trainX, testX, trainY, testY = train_test_split(data2, Y, test_size=0.25, random_state=42)
a = NN.train(trainX,trainY, testX,testY)
print a,testY
# ploting cost vs iteration for train and test set
plt.plot(NN.j)
plt.plot(NN.testj)
plt.show()
plt.legend()

#predicting the error b/w result and actual data for test set
b = a-testY

b
