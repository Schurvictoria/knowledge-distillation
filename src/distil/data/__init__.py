from distil.data.gender import load_gender_dataset
from distil.data.rosbank import load_rosbank_dataset
from distil.data.age import load_age_dataset
from distil.data._downloads import (
    download_gender_data,
    download_rosbank_data,
    download_age_data,
)
from distil.data.embeddings_io import (
    save_coles_embeddings,
    load_coles_embeddings,
)

__all__ = [
    "load_gender_dataset",
    "load_rosbank_dataset",
    "load_age_dataset",
    "download_gender_data",
    "download_rosbank_data",
    "download_age_data",
    "save_coles_embeddings",
    "load_coles_embeddings",
]
