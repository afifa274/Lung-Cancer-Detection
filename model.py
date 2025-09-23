import pandas as pd
import matplotlib.pyplot as plt
import pickle


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

df = pd.read_csv('cancer patient data sets.csv')

level_mapping = {'Low': float(0), 'Medium': float(1.0), 'High': float(2.0)}

df['Level'] = df['Level'].replace(level_mapping)

df.to_csv('cancer patient data sets.csv', index=False)

X = df.iloc[:, 2:-1].values  
y = df.iloc[:, -1].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=23))
model.add(Dense(16, activation='relu'))
model.add(Dense(3, activation='sigmoid'))


optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train section
history = model.fit(X_train, keras.utils.to_categorical(y_train, 3), epochs=10, batch_size=32, validation_data=(X_test, keras.utils.to_categorical(y_test, 3)), verbose=2)


loss, accuracy = model.evaluate(X_test, keras.utils.to_categorical(y_test, 3))
print('Test accuracy:', round(accuracy*100, 2), '%')


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


model.save('cancer_snus_model.h5')


filename = 'scaler.pkl'
with open(filename, 'wb') as f:
    pickle.dump(scaler, f)
