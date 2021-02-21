import pandas as pd
from collections import defaultdict
import argparse

def get_df(conll):
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
    return(df)

def pos_extraction(conll):
    df = get_df(conll)
    pos = defaultdict(int)
    total_pos = 0
    num_of_words = 0

    (rows, columns) = df.shape
    for row in range(rows):
        if not pd.isnull(df.iloc[row, 2]):
            POS = df.iloc[row, 2]
            pos[POS] += 1
            num_of_words += 1
        
    list_of_pos = list(pos.keys())
    perc_pos = dict()
    for POS in pos:
        percentage = pos[POS] / num_of_words
        perc_pos[POS] = percentage
        
    return perc_pos
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Input the path to a .conll file and an output directory")
    parser.add_argument("--conll", dest="conll", type=str)
    parser.add_argument("--output_dir", dest="output_dir", type=str)

    args = parser.parse_args()

    perc_pos = pos_extraction(args.conll)

    with open(args.output_dir, "w") as f:
        for POS in perc_pos:
            perc = perc_pos[POS]
            f.write("%s\t%.2f %%\n" % (POS, perc))