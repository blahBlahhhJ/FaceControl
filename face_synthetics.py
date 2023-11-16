import pandas as pd
import datasets
import os

_VERSION = datasets.Version("2.14.6")

_DESCRIPTION = "TODO"
_HOMEPAGE = "TODO"
_LICENSE = "TODO"
_CITATION = "TODO"

_FEATURES = datasets.Features(
    {
        "image": datasets.Image(),
        "conditioning_image": datasets.Image(),
        "text": datasets.Value("string"),
    },
)

_DEFAULT_CONFIG = datasets.BuilderConfig(name="default", version=_VERSION)


class FaceSynthetics(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [_DEFAULT_CONFIG]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=_FEATURES,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        metadata_train_path = os.path.join(self.config.data_dir, "metadata_train.jsonl")
        metadata_val_path = os.path.join(self.config.data_dir, "metadata_val.jsonl")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "metadata_path": metadata_train_path,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "metadata_path": metadata_val_path,
                },
            ),
        ]

    def _generate_examples(self, metadata_path):
        metadata = pd.read_json(metadata_path, lines=True)

        for _, row in metadata.iterrows():
            id_ = f"{row['id']:06d}"
            text = row["text"]

            image_path = os.path.join(self.config.data_dir, f"{id_}.png")
            conditioning_image_path = os.path.join(self.config.data_dir, f"{id_}_cond.png")

            with open(image_path, "rb") as f:
                image = f.read()
            with open(conditioning_image_path, "rb") as f:
                conditioning_image = f.read()

            yield id_, {
                "text": text,
                "image": {
                    "path": image_path,
                    "bytes": image,
                },
                "conditioning_image": {
                    "path": conditioning_image_path,
                    "bytes": conditioning_image,
                },
            }