# Neural Networks Project WiSe2020/21
This is a project in which we want to train a model to do automatic POS tagging, based on sample data.

## Table of Contents
* <i>sample.info</i>: contains information about the sample data set
* <i>environment.yaml</i>: `.yaml` file to build the environment
* <i>data_preprocessing.py</i>: python program for the preprocessing, creating a `.info` and a `.tsv` file 
* <i>to_df.py</i>: python program splitting a `.tsv` file - for example, the one created by `data_preprocessing.py` - into training, testing, and validating data
* <i>loading_script.py</i>: python program for loading the training, testing and validating data
* <i>create_embeddings.py</i>: python program for tokenization and creating embeddings from the training, testing and validating data using the BERT model
* <i>classifiers.py</i>: python program for the two classifiers CNN and LSTM
* <i>task2.py</i>: python program that will combine all pieces of code, train, test and evaluate the classifiers from `classifiers.py`

## General Information
This project creates a classifier by using BERT-embeddings. We compare the performance of the two classifying algorithms CNN and LSTM. Currently, the code is divided into multiple sub-programs, as mentioned below. Later, however, it is planned that everything will be unified in `task2.py`, so that the user only has to run one line of code to execute the whole project.

## Data Preprocessing
Done using the file `data_preprocessing.py`. After opening the command line, move to the directory containing the program `data_preprocessing.py`. Into the command line, enter
`python data_preprocessing.py --conll=<Path_To_Conll_File> --output_dir=<Path_To_Output_Directory>`. <br>
Press enter. Once the program is finished, you will find the extracted POS tags (as a `.tsv` file) and information (as a `.info` file) about the data set in the specified output directory.

## Splitting the data into training, testing, and validating data
Done using the file `to_df.py`. After opening the command line, move to the directory containing the program `to_df.py`. Into the command line, enter `python to_df.py --tsv=<Path_To_TSV_File> --output_dir=<Path_To_Output_File>`. The program defaults to a `.tsv` file called `sequences.tsv` and the working directory as an output directory. <br>
Press enter. Once the program is finished, you will find the training, testing and validating data in the specified output directory. <br>
The data is split into roughly 70% training, 15% testing and 15% validating data, while also ensuring that no sentences are split up between them. The output takes the form of three lists per row, with each row containing the following information about a sentence:
* `position`: these are the indices for each word of the sentence. For example, a sentence with 3 words will have the following list in `position`: `[0, 1, 2]`
* `word`: this list contains each word of the sentence
* `POS`: this list contains the POS-tag of each word

## Loading the data
Done using the file `loading_script.py`. This is code that does not need to be executed manually, but will be used by the following steps. The function `_split_generators` loads the training, testing and validating data, and the function `_generate_examples` yields examples from these for further use. 

## Creating embeddings
Done using `create_embeddings.py`. The function `load_datasets(dataset)` loads the previously created training, testing and validating data by accessing the defined functions in `loading_script.py`. The function `encode_dataset` tokenizes and creates embeddings from tokenized data using the pre-trained BERT model. <br>
Simply run this program by entering `python create_embeddings.py` into the command line. Once complete, this will later be automatically executed in `task2.py`.

## Classifying algorithms
Done using `classifiers.py`. This code is still unfinished and does not work at this point, and the goal is that it will be automatically executed in `task2.py`.
### CNN
In `classifiers.py`, the class `TextClassifier` is responsible for the CNN classifying algorithm.
### LSTM
In `classifiers.py`, the class `LSTM` is responsible for the LSTM classifying algorithm.

## Executing the algorithms
As of now, each part of the code needs to be executed seperately, as `task2.py` is still unfinished. Read previous sections on how to execute each part. <br>
The goal is for `task2.py` to combine all parts of the code, such that only one command needs to be executed from the command line.
