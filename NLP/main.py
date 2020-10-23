from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten
from keras.callbacks import EarlyStopping
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence
# from keras.utils import plot_model

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import heapq

import string

def model():
    pass

def pre_processing(column):
    """
    removing html tags, punctuations and numbers, multiple spaces
    """
    list = [character for character in column]
    #removing html tags
    try:
        for i in range(len(list)):
            if list[i] == '<':
                #remove this character all characters after till '>' is reached
                while list[i] != ">":
                    list.pop(i)
                list[i] = " "
    except:
        #end of string reached
        pass
    list = [p for p in list if p not in string.punctuation]
    list = [d for d in list if d not in string.digits]
    try:
        for i in range(len(list)):
            if list[i] == ' ' and list[i+1] == " ":
                #remove this character all characters after till '>' is reached
                while list[i+1] == " ":
                    list.pop(i)
    except:
        # End of string reached
        pass
    return "".join(list)
    return list

def counter(tokens):
    wordfreq = {}
    for i in tokens:
        if i[0] not in wordfreq.keys():
            wordfreq[i[0]] = 1
        else:
            wordfreq[i[0]] += 1
    # most_freq = heapq.nlargest(200, wordfreq, key=wordfreq.get)
    return wordfreq

# def feature_selector(features):
#     #features is a dictionary
#     list = []
#     for i in range(0,100):
#         list.append(features[i])
#     return list

def input_convert(tokens):
    list = []
    try:
        for i in range(0,100):
            list.append(tokens[i][0])
    except:
        pass
    return list

if __name__ == "__main__":
    MAX_FEATURES = 200000
    MAX_LEN = 80

    #load and preprocess data
    file = pd.read_csv("/Users/heizer/Desktop/CDS_Lab/Week_6/IMDB_Dataset.csv")
    file['review'] = file['review'].apply(lambda x: pre_processing(x))
    file['sentiment'] = file['sentiment'].apply(lambda x: 1 if x=='positive' else 0)

    tokenizer = Tokenizer(num_words= None, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True,split=' ', char_level=False, oov_token=None, document_count=0)
    file['sequences'] = file['review'].apply(lambda x: text_to_word_sequence(x))
    tokenizer.fit_on_texts(file.sequences)
    file['tokenized'] = file['sequences'].apply(lambda x: tokenizer.texts_to_sequences(x))
    file['input'] = file['tokenized'].apply(lambda x: input_convert(x))
    # file['token_dict'] = file['tokenized'].apply(lambda x: counter(x))
    # file['features'] = file['token_dict'].apply(lambda x: feature_selector(x))

#%%

    #split the data into train and test
    print("Preparing the data...")
    train, test = train_test_split(file, test_size=0.2)

    X_train = train.input
    X_test = test.input
    X_train = sequence.pad_sequences(X_train, maxlen=MAX_LEN)
    X_test = sequence.pad_sequences(X_test, maxlen=MAX_LEN)

    y_train = train.sentiment
    y_test = test.sentiment

    # from sequences 1) remove stop words, 2) lemmatize, 3) replace rare words

    #bag of words representation needed to feed in top 100 vocab words into the neural network

    #defining the model
    model = Sequential()
    #layer 1
    model.add(Embedding(MAX_FEATURES, 128,input_shape=(MAX_LEN,),trainable=True))
    model.add(Flatten())
    #layer 2
    model.add(Dense(128, activation='relu'))
    #layer 3
    model.add(Dense(1, activation='sigmoid'))
    print(model.summary())

    # plot_model(model, to_file='model.png')
    es = EarlyStopping(monitor='val_loss', mode='min', patience=10)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print('Training in progress...')
    model.fit(X_train, y_train, batch_size=32, epochs=15, validation_split=0.1, callbacks=[es])

    #metrics calculation
    score, acc = model.evaluate(X_test, y_test, batch_size=32)
    print('Test score:', score)
    print('Test accuracy:', acc)

