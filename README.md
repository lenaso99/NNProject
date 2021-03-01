# Neural Networks Project WiSe2020/21
This is a project in which we want to train a model to do automatic POS tagging, based on sample data.

## Table of Contents
* <i>sample.info</i>: contains information about the sample data set
* <i>data_preprocessing.py</i>: python program for the preprocessing, creating a `.info` and a `.tsv` file 
* <i>environment.yaml</i>: `.yaml` file to build the environment

## General Information

## Data Preprocessing
Done using the file `data_preprocessing.py`. After opening the command line, move to the directory containing the program `data_preprocessing.py`. Into the command line, enter
`python data_preprocessing.py --conll=Path_To_Conll_File --output_dir=Path_To_Output_File`
Press enter. Once the programm is finished, you will find the extracted POS tags in the specified output file.
