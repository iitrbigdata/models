
# coding: utf-8

# In[1]:

class ANNmodel:
    def __init__(self):
        pass
    def model(self,X,Y):
        from keras.models import Sequential
        from keras.layers import Dense
        # create model
        model = Sequential()
        model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
        model.add(Dense(8, init='uniform', activation='relu'))
        model.add(Dense(1, init='uniform', activation='sigmoid'))
        
        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        # Fit the model
        model.fit(X, Y, nb_epoch=150, batch_size=10, verbose=False)
        
        # evaluate the model
        scores = model.evaluate(X, Y)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# In[ ]:



