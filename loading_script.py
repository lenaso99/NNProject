
# Lint as: python3
"""NN_project"""

from __future__ import absolute_import, division, print_function
import csv
import os
import datasets

_CITATION = """\
Lena Sophie Oberkircher
Leonie Servas
}
"""

_DESCRIPTION = """\
Neural Networks Project: Comparing LSTM with Bert embeddings to CNN
with Bert embeddings
"""

# _DATA_URL = "https://nlp.stanford.edu/projects/snli/snli_1.0.zip"


class NN_project(datasets.GeneratorBasedBuilder):
    """NN Project."""

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="tsv",
            version=datasets.Version("1.0.0", ""),
            #TODO: fill in description
            description="Comparing LSTM to CNN with Bert embeddings",
        )
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "word": datasets.Value("string"),
                    "tag": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage="https://github.com/lenaso99/NNProject",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        data_dir = os.getcwd()
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"filepath": os.path.join(data_dir, "test.tsv")}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, gen_kwargs={"filepath": os.path.join(data_dir, "validate.tsv")}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"filepath": os.path.join(data_dir, "train.tsv")}
            ),
        ]

    def _generate_examples(self, filepath):
        """This function returns examples from tsv data."""
        with open(filepath, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)

            for idx, row in enumerate(reader):

                    yield idx, {
                    "id": row["position"],
                    "word": row["word"],
                    "tag": row["POS"],
                    }
