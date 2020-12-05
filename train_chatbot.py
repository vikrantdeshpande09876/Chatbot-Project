import random, nltk, json, pickle, numpy as np, tensorflow.keras as keras
from pprint import pprint
from nltk.stem import WordNetLemmatizer

lemmatizer=WordNetLemmatizer()
words, classes, docs = [], [], []
ignore_words=['?','!']

with open('intents.json') as f:
    intents=json.loads(f.read())

pprint(intents)


for intent in intents['intents']:
    for pattern in intent['patterns']:
        w=nltk.word_tokenize(pattern)
        words.extend(w)
        
        docs.append((w, intent['tag']))
        
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

print(words)

lem_vocab=sorted(list(set([lemmatizer.lemmatize(w) for w in words if w not in ignore_words])))
classes=sorted(list(set(classes)))

pickle.dump(lem_vocab,open('lem_vocab.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

training=[]
output_empty=[0]*len(classes)

for doc in docs:
    bag=[]
    pattern_words=[lemmatizer.lemmatize(w.lower()) for w in doc[0]]
    for v in lem_vocab:
        bag.append(1) if v in pattern_words else bag.append(0)
    output_row=list(output_empty)
    output_row[classes.index(doc[1])]=1
    training.append([bag,output_row])



random.shuffle(training)
train_df=np.array(training)
train_x=list(train_df[:,0])
train_y=list(train_df[:,1])

print('\n\n{} TRAIN DATASET PREPROCESSING DONE {} \n\n'.format('#'*16,'#'*16))


model=keras.Sequential()
model.add(keras.layers.Input(shape=(len(train_x[0]),)))
model.add(keras.layers.Dense(units=128, activation='relu'))
model.add(keras.layers.Dropout(rate=0.5))
model.add(keras.layers.Dense(units=64, activation='relu'))
model.add(keras.layers.Dropout(rate=0.5))
model.add(keras.layers.Dense(units=len(train_y[0]), activation='softmax'))

sgd=keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist=model.fit(x=np.array(train_x), y=np.array(train_y), epochs=200, batch_size=8, verbose=1)
model.save('chatbot.h5',hist)

print('\n\n{} MODEL CREATION DONE {} \n\n'.format('#'*16,'#'*16))
