import pandas as pd
from collections import defaultdict
import argparse
import numpy as np

def get_df(tsv, length):
    df = pd.read_csv(tsv, sep="\t")
    train = pd.DataFrame(columns = ["position", "word", "POS"])
    test = pd.DataFrame(columns = ["position", "word", "POS"])
    validate = pd.DataFrame(columns = ["position", "word", "POS"])

    training = True
    testing = False
    validating = False

    (rows, columns) = df.shape
    for row in range(rows):
        position = df.iloc[row]["position"]
        word = df.iloc[row]["word"]
        pos = df.iloc[row]["POS"]

        if training:
            train = train.append({"position": position, "word": word, "POS": pos}, ignore_index=True)
            if row/length > 0.7 and position == "*":
                training = False
                testing = True

        elif testing:
            test = test.append({"position": position, "word": word, "POS": pos}, ignore_index=True)
            if row/length > 0.85 and position == "*":
                testing = False
                validating = True

        elif validating:
            validate = validate.append({"position": position, "word": word, "POS": pos}, ignore_index=True)


    # df.to_csv("seq.tsv", sep="\t", index=False)
    train.to_csv("train.tsv", sep="\t", index=False)
    test.to_csv("test.tsv", sep="\t", index=False)
    validate.to_csv("validate.tsv", sep="\t", index=False)
    return(df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Input the path to a .tsv file and an output directory for the output files.")
    parser.add_argument("--tsv", dest="tsv", type=str, default="sequences.tsv")
    parser.add_argument("--output_dir", dest="output_dir", type=str, default=".")

    args = parser.parse_args()

    length = 0
    with open(args.tsv) as g:
        for l in g:
            length += 1

    get_df(args.tsv, length)
