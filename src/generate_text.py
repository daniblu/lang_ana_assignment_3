import os, argparse, pickle
import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# functions
def input_parse():
    """Allows arguments to be passed to the script form the terminal"""
    # initialize the parser
    parser = argparse.ArgumentParser(description="Generate text from prompt using trained model")
    # add arguments
    parser.add_argument("--id", type=int, required=True, help="id of model to load")
    parser.add_argument("--prompt", type=str, nargs='+', required=True, help="start of sentence to be continued by the model")
    parser.add_argument("--n", type=int, required=True, help="number of words to predict following the prompt")
    #parse the arguments from the terminal
    args = parser.parse_args()

    return args

def generate_text(seed_text, next_words, model, tokenizer, max_sequence_len):
    """Preprocesses seed_text so that it is compatible with model and iteratively predicts next words"""
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], 
                                    maxlen=max_sequence_len-1, 
                                    padding='pre')
        predicted = np.argmax(model.predict(token_list, verbose=0),
                                            axis=1)
        
        output_word = ""
        for word,index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " "+output_word
    return seed_text.title()

def main(id, prompt, n):
    
    # load model
    modelPath = os.path.join("..", "models", f"model{id}.h5")
    model = tf.keras.models.load_model(modelPath)

    # load tokenizer and max_sequence_len
    tokenizerPath = os.path.join("..", "models", "preprocessing_objects.pkl")
    with open(tokenizerPath, 'rb') as file:
        tokenizer, max_sequence_len = pickle.load(file)
    
    # restructure prompt (a list) into string
    seed_text = ' '.join(prompt)

    # generate text and print
    print(generate_text(seed_text, n, model, tokenizer, max_sequence_len))

if __name__ == "__main__":
    args = input_parse()
    main(args.id, args.prompt, args.n)