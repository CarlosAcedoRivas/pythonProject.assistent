import nltk
nltk.download('punkt_tab')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.optimizers import SGD
from keras.optimizers.schedules import ExponentialDecay
import random
from PIL import Image

data_file = open('intents_script_preguntesirespostes.json', 'r', encoding='utf-8').read()
intents = json.loads(data_file)

lemmatizer = WordNetLemmatizer()

words=[]
classes = []
documents = []
ignore_words = ['?', '!']

# Recorre cada intenció i els seus patrons a l'arxiu JSON
for intent in intents['intents']:
    for pattern in intent['patterns']:
        #Tokenitza les paraules a cada patró i les afegeix a la llista de paraules
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        #Afegeix el parell (patró, etiqueta) a la llista de documents
        documents.append((w, intent['tag']))
        #Si l'etiqueta no està al llistat de classes, la afegeix
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lematitza les paraules i les converteix em minúscules, excluïnt les paraules ignorades
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

#Desa els llistats de paraules i classes en arxius pickle
pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

training = []
output_empty = [0] * len(classes)

#Crea el conjunt d'entrenament
for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    for word in words:
        #Crea una bossa de paraules binària per a cada patró
        bag.append(1) if word in pattern_words else bag.append(0)
    output_row = list(output_empty)
    #Crea un vector de sortida amb un 1 a la possició corresponent a l'etiqueta de la intenció
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

#Barreja aleatoriament el conjunt d'entrenament
random.shuffle([bag, output_row])

#Divideix el conjunt d'entrenament en característiques (train_x) i etiquetes (train_y)
train_x = [row[0] for row in training]
train_y = [row[1] for row in training]

train_x = np.array(train_x)
train_y = np.array(train_y)

#Crea el model de xarxa neuronal
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0,5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0,5))
model.add(Dense(len(train_y[0]), activation='softmax'))

#Configura l'optimitzador amb una taxa d'aprenentatge exponencialment decreixent
lr_schedule = ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=10000,
    decay_rate=0.9)

sgd = SGD(learning_rate=lr_schedule, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#Entrena el model amb el conjunt d'entrenament
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

#Desa el model entrenat en un arxiu h5
model.save('chatbot_model.h5', hist)

print("model created")