from keras.models import Sequential

from keras.layers import Dense

import numpy
numpy.random.seed(7)
X=Final_dataset[['Casualty_Type', 'Road_Type', 'Light_Conditions', 'Weather_Conditions', 'Junction_Location', 
                               'Age_of_Driver','Accident_Severity','Age_of_Vehicle']]
Y=Final_dataset[['Sex_of_Casualty']]
model = Sequential()

model.add(Dense(12, input_dim=8, activation='relu'))

model.add(Dense(8, activation='relu'))

model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, Y, epochs=150, batch_size=10)
