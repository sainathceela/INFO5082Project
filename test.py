import os
import FeaturesData
import numpy as np
import pandas as pd
'''
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import pickle
import os
from keras.utils.np_utils import to_categorical
'''
from features import PEFeatureExtractor
file_data = open('antialias.exe', "rb").read()
extractor = PEFeatureExtractor(2)
features = np.array(extractor.feature_vector(file_data), dtype=np.float32)
print(features)
print(features.shape)
'''
N = 50000
dim = 2381
data_dir = "C:/acc/EagerClient/data1/ember2018/"
X, Y = FeaturesData.read_vectorized_features(data_dir,"train")
X = X[0:5000]
Y = Y[0:5000]
Y = to_categorical(Y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1, 1))
print(Y)
print(Y.shape)
print(X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,random_state = 0)
if os.path.exists('model/model.json'):
    with open('model/model.json', "r") as json_file:
        loaded_model_json = json_file.read()
        classifier = model_from_json(loaded_model_json)
    classifier.load_weights("model/model_weights.h5")
    classifier._make_predict_function()   
    print(classifier.summary())
    f = open('model/history.pckl', 'rb')
    data = pickle.load(f)
    f.close()
    acc = data['accuracy']
    accuracy = acc[9] * 100
    print("Training Model Accuracy = "+str(accuracy))
else:
    classifier = Sequential()
    classifier.add(Convolution2D(32, 1, 1, input_shape = (X.shape[1], 1, 1), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (1, 1)))
    classifier.add(Convolution2D(32, 1, 1, activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (1, 1)))
    classifier.add(Flatten())
    classifier.add(Dense(output_dim = 256, activation = 'relu'))
    classifier.add(Dense(output_dim = Y.shape[1], activation = 'softmax'))
    print(classifier.summary())
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    hist = classifier.fit(X, Y, batch_size=16, epochs=10, shuffle=True, verbose=2)
    classifier.save_weights('model/model_weights.h5')            
    model_json = classifier.to_json()
    with open("model/model.json", "w") as json_file:
        json_file.write(model_json)
    f = open('model/history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()
    f = open('model/history.pckl', 'rb')
    data = pickle.load(f)
    f.close()
    acc = data['accuracy']
    accuracy = acc[9] * 100
    print("Training Model Accuracy = "+str(accuracy))

'''
''''
rfc = RandomForestClassifier(random_state=0)
rfc.fit(X_train, y_train)
predict = rfc.predict(X_test) 
random_acc = accuracy_score(y_test,predict)*100
print(random_acc)

un,count = np.unique(Y, return_counts=True)
print(un)
print(count)
'''

'''
def prepareFeatures():
    data_dir = "C:/acc/EagerClient/data1/ember2018/"
    raw_feature_paths = [os.path.join(data_dir, "train_features_{}.jsonl".format(i)) for i in range(1)]
    print(raw_feature_paths)
    FeaturesData.create_vectorized_features(data_dir)
    _ = FeaturesData.create_metadata(data_dir)
    emberdf = FeaturesData.read_metadata(data_dir)
    X_train, y_train, X_test, y_test = FeaturesData.read_vectorized_features(data_dir)
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)


if __name__ == '__main__':
    prepareFeatures()

'''
