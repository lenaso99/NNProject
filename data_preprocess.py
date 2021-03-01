import pandas as pd
from collections import defaultdict
import argparse

def get_df(conll):
    # Extracts the data frame from the .conll file.
    df = pd.DataFrame(columns = ["position", "word", "POS"])
    with open(conll) as f:
        for line in f:
            if not line.startswith("#") and len(line)>2:
                line = ' '.join(line.split())
                content = line.split()
                position = content[2]
                word = content[3]
                pos = content[4]
                new_row = {"position": position, "word": word, "POS": pos}
                df = df.append(new_row, ignore_index = True)
                
            elif len(line) <= 2:
                new_row = {"position": "*", "word": "", "POS": ""}
                df = df.append(new_row, ignore_index = True)
    
    # Create the .tsv file containing the sequences, one word and POS tag at a time.
    df.to_csv("sequences.tsv", sep="\t", index=False)
    return(df)

def pos_extraction(conll):
    # Extracts percentages of the POS tags from the dataframe. For the .info file.
    df = get_df(conll)
    pos = defaultdict(int)
    num_of_words = 0

    (rows, columns) = df.shape
    for row in range(rows):
        if not pd.isnull(df.iloc[row, 2]):
            POS = df.iloc[row, 2]
            pos[POS] += 1
            num_of_words += 1
        
    perc_pos = dict()
    for POS in pos:
        percentage = pos[POS] / num_of_words
        perc_pos[POS] = percentage
    return perc_pos

def sequences(conll):
    # Extracts info about sequence lengths etc. Also for the .info file.
    df = get_df(conll)

    num_of_sequences = 1
    seq_lengths = list()
    
    (rows, columns) = df.shape

    for row in range(rows):
        if df.at[row, 0] == "*":
            num_of_sequences += 1
            length = (int(df.iloc[row-1, 0]) + 1)
            seq_lengths.append(length)
    length = (int(df.iloc[rows, 0]) + 1)
    seq_lengths.append(length)
    max_seq_length = max(seq_lengths)
    min_seq_length = min(seq_lengths)
    avg_seq_length = (sum(seq_lengths) / len(seq_lengths))

    seq_info = [max_seq_length, min_seq_length, avg_seq_length, num_of_sequences]
    return seq_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Input the path to a .conll file and an output directory for the .info file.")
    parser.add_argument("--conll", dest="conll", type=str)
    parser.add_argument("--output_dir", dest="output_dir", type=str)

    args = parser.parse_args()

    perc_pos = pos_extraction(args.conll)

    seq = sequences(args.conll)
    max_seq_length = seq[0]
    min_seq_length = seq[1]
    avg_seq_length = seq[2]
    num_of_sequences = seq[3]
    with open(args.output_dir, "w") as f:
        # Creates the .info file.
        f.write("Max sequence length: %s\n" % (max_seq_length))
        f.write("Min sequence length: %s\n" % (min_seq_length))
        f.write("Mean sequence length: %.2f\n" % (avg_seq_length))
        f.write("Number of sequences: %s\n\n" % (num_of_sequences))

        f.write("Tags:\n")
        for POS in perc_pos:
            perc = perc_pos[POS]
            f.write("%s\t%.2f %%\n" % (POS, perc))
