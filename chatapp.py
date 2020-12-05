import random, nltk, json, pickle, numpy as np, tensorflow.keras as keras
from pprint import pprint
from nltk.stem import WordNetLemmatizer


def return_cleaned_sentence(sentence):
    return list([lemmatizer.lemmatize(w) for w in nltk.word_tokenize(sentence) if w not in ignore_words])


def bag_of_words(sentence, lem_vocab, verbose=True):
    lem_sentence=return_cleaned_sentence(sentence)
    bag=[0]*len(lem_vocab)
    for s in lem_sentence:
        for ind, v in enumerate(lem_vocab):
            if s==v:
                bag[ind]=1
                if verbose:
                    print('\nFound {} in bag of words at index {}\n'.format(s,ind))
    return np.array(bag)

def predict_class(model, sentence):
    bag=bag_of_words(sentence, lem_vocab, verbose=False)
    assert np.array([bag]).shape[1] == model.input.shape[1]
    res=model.predict(np.array([bag]))[0]
    ERR_THRESHOLD=.25
    results=[[i,pr] for i,pr in enumerate(res) if pr>=ERR_THRESHOLD]
    results.sort(key=lambda x:x[1], reverse=True)
    result_list=[]
    for r in results:
        result_list.append({'tag':classes[r[0]], 'prob':str(r[1])})
    return result_list
    
    
def get_response(preds, intents_json):
    best_tag=preds[0]['tag']
    list_of_intents=intents_json['intents']
    for intent in list_of_intents:
        if intent['tag']==best_tag:
            result=random.choice(intent['responses'])
            break
    return result


def chatbot_response(text):
    preds=predict_class(model, text)
    return get_response(preds, intents)
    
    
if __name__=='__main__':
    flag=True
    lemmatizer=WordNetLemmatizer()
    intents=json.loads(open('intents.json','rb').read())
    lem_vocab=pickle.load(open('lem_vocab.pkl','rb'))
    classes=pickle.load(open('classes.pkl','rb'))
    ignore_words=['?','!']
    model=keras.models.load_model('chatbot.h5')
    print('\n\n{} READING PICKLE OBJECTS DONE {} \n\n'.format('#'*16,'#'*16))
    name=input("What's your name?\n ")
    
    while True:
        text=input('\n\n{} (Type N to exit)>\t'.format(name))
        if text.lower() in ['n','no','exit','quit','q']:
            flag=False
            break
        else:
            print('Chatbot>\t',chatbot_response(text))