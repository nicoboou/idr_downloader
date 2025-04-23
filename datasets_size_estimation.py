import argparse
import glob
import logging
import os
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from tqdm import tqdm

# from src.modules.datadownloaders.IDR.tiny_dataset_downloader import TinyDatasetDownloader
from src.modules.datadownloaders.IDR.data_downloader import DataDownloader

logging.basicConfig(filename="data_size_estimation_error.log", level=logging.ERROR)


@dataclass
class DatasetEstimationParams:
    NUM_IMAGES_REQUESTED = 10
    NORMALIZE_IMAGES = True
    RESIZE_SIZE = 224
    DATA_TYPES = ["cell", "tissue"]


def get_dataframe(data_type: Literal["cell", "tissue"]) -> pd.DataFrame:
    """
    Retrieve the dataframe from the file path
    """
    # enforce data type bheing either cell or tissue
    if data_type not in DatasetEstimationParams.DATA_TYPES:
        raise ValueError(
            f"Invalid data type. Must be in {DatasetEstimationParams.DATA_TYPES} "
        )
    try:
        dataframe = pd.read_csv(f"data/IDR/{data_type}/full_metadata_dataframe.csv")
        return dataframe
    except Exception as e:
        logging.exception(f"An error occurred while reading the dataframe: {e}")
        return None


def get_metadata_files(dataframe: pd.DataFrame) -> dict:
    """
    Retrieve the metadata from the file path
    """
    try:
        # Create disctionary of imagging method for each metadata file
        metadata_files = dataframe["metadata_file"].unique().tolist()
        # metadata_files = ['metadata_experiment_401.json']
        return metadata_files
    except Exception as e:
        logging.exception(
            f"An error occurred while retrieving the metadata filepaths: {e}"
        )
        return None


def estimate(
    data_type: Literal["cell", "tissue"],
    dataframe: pd.DataFrame,
    metadata_files: list,
    metadata_filepath: str,
):

    for metadata_file in tqdm(metadata_files):
        try:
            metadata_filepath = os.path.join(
                "data", "IDR", data_type, "meta", metadata_file
            )
            print(f"Estimation for metadata file: {metadata_file}... ")
            data_downloader = DataDownloader(metadata_filepath=metadata_filepath)

            # ---------------------------------------- ORIGINAL Data Size Estimation ---------------------------------------- #
            # Run this cell to download the dataset
            dataset_dict, random_image_ids_selected, images_resolution_dict = (
                data_downloader.download(
                    num_images_requested=DatasetEstimationParams.NUM_IMAGES_REQUESTED,
                    normalize=DatasetEstimationParams.NORMALIZE_IMAGES,
                )
            )

            for experiment_id, experiment_images in dataset_dict.items():
                print(experiment_id)
                # If the experiment is not empty, let's save it
                if bool(experiment_images):
                    # Retrieve the list of images
                    experiment_images_list, experiment_image_ids = list(
                        experiment_images.values()
                    ), list(experiment_images.keys())

                    # Filenames
                    filename_npz = (
                        f"data/IDR/{data_type}/samples/images_npz/"
                        + "experiment_"
                        + str(experiment_id)
                        + ".npz"
                    )
                    filename_png = (
                        f"data/IDR/{data_type}/samples/images_png/original/"
                        + "experiment_"
                        + str(experiment_id)
                    )

                    # Save images
                    DataDownloader.save_to_npz(
                        images=experiment_images_list,
                        labels=experiment_image_ids,
                        filename=filename_npz,
                    )
                    DataDownloader.save_to_png(
                        images=experiment_images_list,
                        labels=experiment_image_ids,
                        filename=filename_png,
                    )

                    dataframe = pd.read_csv(
                        f"data/IDR/{data_type}/full_metadata_dataframe.csv"
                    )

                    # ---------------- Get file size NPZ ---------------- #
                    file_size_npz = os.path.getsize(filename_npz)
                    file_size_npz = (
                        file_size_npz / DatasetEstimationParams.NUM_IMAGES_REQUESTED
                    )
                    dataframe.loc[
                        dataframe["experiment_id"] == experiment_id,
                        "average_image_size_npz_original",
                    ] = file_size_npz

                    # ---------------- Get file size PNG ---------------- #
                    file_sizes_png = []

                    # Find all files matching the pattern
                    for img_id in experiment_image_ids:
                        channels_sizes_png = []

                        filename_png = (
                            f"data/IDR/{data_type}/samples/images_png/original/"
                            + "experiment_"
                            + str(experiment_id)
                            + "_"
                            + str(img_id)
                        )
                        file_list_png = glob.glob(filename_png + "*.png")

                        for channel_img in file_list_png:
                            # Get the size of the channel file in bytes
                            channel_file_size = os.path.getsize(channel_img)
                            channels_sizes_png.append(channel_file_size)

                        file_sizes_png.append(np.sum(channels_sizes_png))

                    average_file_size = np.mean(file_sizes_png)
                    dataframe.loc[
                        dataframe["experiment_id"] == experiment_id,
                        "average_image_size_png_original",
                    ] = average_file_size

                    # Delete images
                    print("Deleting images...")
                    os.remove(filename_npz)
                    for file in glob.glob(
                        f"data/IDR/{data_type}/samples/images_png/original/"
                        + "experiment_"
                        + str(experiment_id)
                        + "*.png"
                    ):
                        os.remove(file)
                    print("Images deleted.")

                    # ---------------- Append unique resolutions ---------------- #
                    unique_resolutions = list(
                        set(images_resolution_dict[experiment_id])
                    )
                    dataframe.loc[
                        dataframe["experiment_id"] == experiment_id, "resolutions"
                    ] = str(unique_resolutions)

                    dataframe.to_csv(
                        f"data/IDR/{data_type}/full_metadata_dataframe.csv", index=False
                    )
                    print("Dataframe saved.")

            # ---------------------------------------- RESIZED Data Size Estimation ---------------------------------------- #
            # Run this cell to download the dataset
            dataset_dict, random_image_ids_selected, images_resolution_dict = (
                data_downloader.download(
                    num_images_requested=DatasetEstimationParams.NUM_IMAGES_REQUESTED,
                    normalize=DatasetEstimationParams.NORMALIZE_IMAGES,
                    resize_size=DatasetEstimationParams.RESIZE_SIZE,
                )
            )

            for experiment_id, experiment_images in dataset_dict.items():
                # If the experiment is not empty, let's save it
                if bool(experiment_images):
                    # Retrieve the list of images
                    experiment_images_list, experiment_image_ids = list(
                        experiment_images.values()
                    ), list(experiment_images.keys())

                    # Filenames
                    filename_npz = (
                        f"data/IDR/{data_type}/samples/images_npz/"
                        + "experiment_"
                        + str(experiment_id)
                        + "_resized.npz"
                    )
                    filename_png = (
                        f"data/IDR/{data_type}/samples/images_png/resized/"
                        + "experiment_"
                        + str(experiment_id)
                        + "_resized"
                    )

                    # Save images
                    DataDownloader.save_to_npz(
                        images=experiment_images_list,
                        labels=experiment_image_ids,
                        filename=filename_npz,
                    )
                    DataDownloader.save_to_png(
                        images=experiment_images_list,
                        labels=experiment_image_ids,
                        filename=filename_png,
                    )

                    dataframe = pd.read_csv(
                        f"data/IDR/{data_type}/full_metadata_dataframe.csv"
                    )

                    # ---------------- Get file size NPZ ---------------- #
                    file_size_npz = os.path.getsize(filename_npz)
                    file_size_npz = (
                        file_size_npz / DatasetEstimationParams.NUM_IMAGES_REQUESTED
                    )
                    dataframe.loc[
                        dataframe["experiment_id"] == experiment_id,
                        "average_image_size_npz_resized_224",
                    ] = file_size_npz

                    # ---------------- Get file size PNG ---------------- #
                    file_sizes_png = []

                    # Find all files matching the pattern
                    for img_id in experiment_image_ids:
                        channels_sizes_png = []

                        filename_png = (
                            f"data/IDR/{data_type}/samples/images_png/resized/"
                            + "experiment_"
                            + str(experiment_id)
                            + "_resized_"
                            + str(img_id)
                        )
                        file_list_png = glob.glob(filename_png + "*.png")

                        for channel_img in file_list_png:
                            # Get the size of the channel file in bytes
                            channel_file_size = os.path.getsize(channel_img)
                            channels_sizes_png.append(channel_file_size)

                        file_sizes_png.append(np.sum(channels_sizes_png))

                    average_file_size = np.mean(file_sizes_png)
                    dataframe.loc[
                        dataframe["experiment_id"] == experiment_id,
                        "average_image_size_png_resized_224",
                    ] = average_file_size

                    # Delete images
                    print("Deleting images...")
                    os.remove(filename_npz)
                    for file in glob.glob(
                        f"data/IDR/{data_type}/samples/images_png/resized/"
                        + "experiment_"
                        + str(experiment_id)
                        + "*.png"
                    ):
                        os.remove(file)
                    print("Images deleted.")

                    # ---------------- Append unique resolutions ---------------- #
                    unique_resolutions = list(
                        set(images_resolution_dict[experiment_id])
                    )
                    dataframe.loc[
                        dataframe["experiment_id"] == experiment_id,
                        "resolutions_resized_224",
                    ] = str(unique_resolutions)

                    dataframe.to_csv(
                        f"data/IDR/{data_type}/full_metadata_dataframe.csv", index=False
                    )
                    print("Dataframe saved.")

        except Exception as e:
            # Log the exception if needed
            logging.exception(f"An error occurred for {metadata_file}")
            # Continue to the next iteration of the loop
            continue


def run(data_type: Literal["cell", "tissue"]):
    df = get_dataframe(data_type=data_type)
    metadata_files = get_metadata_files(dataframe=df)
    metadata_filepath = os.path.join("data", "IDR", data_type, "meta")
    estimate(
        data_type=data_type,
        dataframe=df,
        metadata_files=metadata_files,
        metadata_filepath=metadata_filepath,
    )


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--data_type", type=str, default="tissue", choices=["cell", "tissue"]
    )
    args = argparser.parse_args()

    run(data_type=args.data_type)
