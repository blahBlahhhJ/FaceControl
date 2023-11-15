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

DATA_DIR = "./dataset_100"

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
        metadata_path = os.path.join(DATA_DIR, "metadata.jsonl")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "metadata_path": metadata_path,
                },
            ),
        ]

    def _generate_examples(self, metadata_path, images_dir, conditioning_images_dir):
        metadata = pd.read_json(metadata_path, lines=True)

        for _, row in metadata.iterrows():
            id = f"{row['id']:06d}"
            text = row["text"]

            image_path = os.path.join(DATA_DIR, f"{id}.png")
            conditioning_image_path = os.path.join(DATA_DIR, f"{id}_cond.png")

            with open(image_path, "rb") as f:
                image = f.read()
            with open(conditioning_image_path, "rb") as f:
                conditioning_image = f.read()

            yield id, {
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