# language analytics assignment 3
This repository is assignment 3 out of 5, to be sumbitted for the exam of the university course [Language Analytics](https://kursuskatalog.au.dk/en/course/115693/Language-Analytics) at Aarhus Univeristy.

The first section describes the assignment task as defined by the course instructor. The section __Student edit__ is the student's description of how the repository solves the task and how to use it.

# Language modelling and text generation using RNNs

Text generation is hot news right now!

For this assignemnt, you're going to create some scripts which will allow you to train a text generation model on some culturally significant data - comments on articles for *The New York Times*. You can find a link to the data [here](https://www.kaggle.com/datasets/aashita/nyt-comments).

You should create a collection of scripts which do the following:

- Train a model on the Comments section of the data
  - [Save the trained model](https://www.tensorflow.org/api_docs/python/tf/keras/models/save_model)
- Load a saved model
  - Generate text from a user-suggested prompt

## Objectives

Language modelling is hard and training text generation models is doubly hard. For this course, we lack somewhat the computationl resources, time, and data to train top-quality models for this task. So, if your RNNs don't perform overwhelmingly, that's fine (and expected). Think of it more as a proof of concept.

- Using TensorFlow to build complex deep learning models for NLP
- Illustrating that you can structure repositories appropriately
- Providing clear, easy-to-use documentation for your work.

## Student edit
### Solution
The code written for this assignment can be found within the ```src``` directory. The directory contains two scripts with parameter values that can be set from a terminal. _The scripts assume that ```src``` is the working directory_. Here follows a description of the funcionality of each script:

- __model_train.py__: Loads comments data from the ```data``` directory, preprocesses the data and submits it for training of a text generating model. The model is build in ```TensorFlow``` and includes an embedding layer, an LSTM layer, and an output layer using softmax activation. See ```python3 model_train.py -h``` for an overview of manipulatable parameters. Note, model training may last up to several hours. The script outputs several objects all in the ```models``` directory. This includes: a trained model, a text file summarizing the parameters of the model, and ```preprocessing_objects.pkl```, which contains objects to be used for text generation. 

- __generate_text.py__: Loads the trained model and ```preprocessing_objects.pkl``` in order to generate text that continues a prompt passed to the script by the user. See ```python3 geenrate_text.py -h``` for user instructions.

### Results
The model already saved in ```models``` was only trained for one epoch on a small subset of the data, and is simply there to demonstrate the funcionality of this repository. Nevertheless, the model's performance is evaluated by running the following:

```shell
python3 generate_text.py --id 1 --prompt It is time for --n 5
```

This returns: _"It Is Time For The Same Of The House"_

### Setup
The data contains information about comments made on the articles published in New York Times in Jan-May 2017 and Jan-April 2018. The data must be dowloaded from [here](https://www.kaggle.com/datasets/aashita/nyt-comments) and saved in the ```data``` directory.

The scripts require the following to be run from the terminal:

```shell
bash setup.sh
```

This will create a virtual environment, ```assignment3_env``` (git ignored), to which the packages listed in ```requirements.txt``` will be downloaded. __Note__, ```setup.sh``` works only on computers running POSIX. Remember to activate the environment by running the following line in a terminal before changing the working directory to ``src`` and running the ```.py```-scripts.

```shell 
source ./assignment3_env/bin/activate
```
