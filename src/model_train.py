# data processing tools
print("Importing packages")
import string, os, argparse, pickle
import pandas as pd
import numpy as np
np.random.seed(42)

# keras module for building LSTM 
import tensorflow as tf
tf.random.set_seed(42)
import tensorflow.keras.utils as ku 
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.saving import save_model

# functions
def input_parse():
    """Allows arguments to be passed to the script form the terminal"""
    # initialize the parser
    parser = argparse.ArgumentParser(description="Train text generator model")
    # add arguments
    parser.add_argument("--id", type=int, required=True, help="unique int to identify model")
    parser.add_argument("--embedding_length", type=int, default=10, help="length of word embeddings, default=10")
    parser.add_argument("--lstm_units", type=int, default=100, help="number of units in LSTM layer, default=100")
    parser.add_argument("--dropout", type=float, default=0.1, help="proportion of dropout for model training, default=0.1")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs in model training, default=10")
    parser.add_argument("--batch_size", type=int, default=100, help="batch size for model training, default=100")
    #parse the arguments from the terminal
    args = parser.parse_args()
        
    return(args)

def load_data(data_dir):
    """Loads comments from the data directory"""
    all_comments = []
    for filename in os.listdir(data_dir):
        if 'Comments' in filename:
            comments_df = pd.read_csv(os.path.join(data_dir, filename), usecols=["commentBody"])
            # comments are nested in arrays so flatten() is used
            # to obtain the desired structure of a list with comments as entries
            all_comments.extend(list(comments_df.values.flatten()))

    return all_comments

def clean_text(txt):
    """Takes a single comment and cleans it from unwanted characters, changes to lower case and sets encoding"""
    txt = txt.replace("<br/>", " ")
    txt = "".join(v for v in txt if v not in string.punctuation).lower()
    txt = txt.encode("utf8").decode("ascii",'ignore')
    return txt

def get_sequence_of_tokens(tokenizer, corpus):
    """Converts comments to sequences of tokens and breaks them down into increasing n-grams""" 
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
    return input_sequences

def generate_padded_sequences(input_sequences, total_words):
    """Makes all sequences equal length by pre-padding zeros"""
    # get the length of the longest sequence
    max_sequence_len = max([len(x) for x in input_sequences])
    # make every sequence the length of the longest on
    input_sequences = np.array(pad_sequences(input_sequences, 
                                            maxlen=max_sequence_len, 
                                            padding='pre'))
    # last column of input_sequences contains the tokens/labels to be predicted by the foregoing tokens
    predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
    # transform labels to binary matrix
    label = ku.to_categorical(label, 
                            num_classes=total_words)
    return predictors, label, max_sequence_len

def create_model(max_sequence_len, total_words, embedding_length, lstm_units, dropout):
    """Initialises and returns model"""
    input_len = max_sequence_len - 1
    model = Sequential()
    
    # Add Input Embedding Layer
    model.add(Embedding(total_words, 
                        embedding_length, 
                        input_length=input_len))
    
    # Add Hidden Layer 1 - LSTM Layer
    model.add(LSTM(lstm_units))
    model.add(Dropout(dropout))
    
    # Add Output Layer
    model.add(Dense(total_words, 
                    activation='softmax'))

    model.compile(loss='categorical_crossentropy', 
                    optimizer='adam')
    
    return model

def main(id, embedding_length, lstm_units, dropout, epochs, batch_size):
    
    # load data
    print("Loading data")
    dataPath = os.path.join("..", "data")
    all_comments = load_data(dataPath)

    # clean data
    print("Cleaning data")
    corpus = [clean_text(txt) for txt in all_comments]

    # tokenize comments, i.e., rank words according to frequency
    print("Tokenizing data")
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)

    # convert comments to sequence of ranks/tokens
    print("Creating sequences")
    inp_sequences = get_sequence_of_tokens(tokenizer, corpus)
    
    # number of words in vocabulary of comments
    total_words = len(tokenizer.word_index) + 1
   
    # isolate padded sequences for predicting and labels/tokens to be predicted
    predictors, label, max_sequence_len = generate_padded_sequences(inp_sequences, total_words)

    # create model
    model = create_model(max_sequence_len, total_words, embedding_length, lstm_units, dropout)

    # fit model
    print("Training model")
    history = model.fit(predictors, 
                    label, 
                    epochs=epochs,
                    batch_size=batch_size, 
                    verbose=1)

    # save model, model report, tokenizer, and max_sequence_len for text generation
    model.save(os.path.join("..", "models", f"model{id}.h5"))

    txtpath = os.path.join("..", "models", f"model{id}.txt")
    with open(txtpath, "w") as file:
        L = [f"Embedding length: {embedding_length} \n", 
            f"LSTM units: {lstm_units} \n",
            f"Dropout: {dropout} \n",
            f"Epochs: {epochs} \n",
            f"Batch size: {batch_size}"]
        file.writelines(L)

    preprocessing_objects = [tokenizer, max_sequence_len]
    tokenizerPath = os.path.join("..", "models", "preprocessing_objects.pkl")
    with open(tokenizerPath, 'wb') as file:
        pickle.dump(preprocessing_objects, file)

if __name__ == "__main__":
    args = input_parse()
    main(args.id, args.embedding_length, args.lstm_units, args.dropout, args.epochs, args.batch_size)
