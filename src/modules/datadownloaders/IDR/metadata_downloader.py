#!/usr/bin/env python
# coding: utf-8

# Standard imports
import os
import sys
import argparse
import json
import requests
from tqdm import tqdm
import pandas as pd
import multiprocessing

dir_path = os.getcwd()
sys.path.insert(0, dir_path)


class MetaDataDownloader:
    """
    Class to download metadata from IDR.

    Attributes
    ----------
    filename (str): Name of the json file to save the metadata.
    batch_size (int): Number of images to download per batch.
    experiment_id_start (int): Experiment ID to start from.
    nb_experiments (int): Number of screens to download.
    image_metadata (bool): Whether to download image metadata or not.
    data_type (str): Type of screen to download.
    multiprocessing (bool): Whether to use multiprocessing capabilities or not.
    """

    def __init__(
        self,
        experiment_ids,
        request_num_cpus: int = 1,
        filename: str = "metadata.json",
        batch_size: int = 2048,
        image_metadata: bool = False,
        data_type: str = "cell",
    ):
        self.filename = filename
        self.batch_size = batch_size
        self.image_metadata = image_metadata
        self.data_type = data_type
        self.request_num_cpus = request_num_cpus

        self.metadata = {}
        self.session = None
        self.total_nb_images = 0
        self.screens_counter = 0
        self.projects_counter = 0

        self.base_url = "https://idr.openmicroscopy.org"
        self.experiment_types = ["project", "screen"]

        if isinstance(experiment_ids, int):
            self.experiment_ids = [experiment_ids]
        elif isinstance(experiment_ids, list):
            self.experiment_ids = experiment_ids
        else:
            raise ValueError("experiment_ids must be either an integer or a list")

    def download(self):
        """
        Downloads metadata from IDR and saves it in a json file.
        """

        assert self.data_type in [
            "cell",
            "tissue",
            "all",
        ], "Data type must be 'cell', 'tissue' or 'all'"
        assert len(self.experiment_ids) > 0, "nb_experiments must be > 0"

        # Launch session to connect to API
        self.launch_session()

        # Instanciate metadata dict
        self.metadata["screens"] = {}
        self.metadata["projects"] = {}

        print(
            f"Task: Metadata download \n Number of CPUs available: {multiprocessing.cpu_count()} \n Number of CPUs requested: {self.request_num_cpus}"
        )

        # Create a multiprocessing.Pool object with the desired number of processes
        pool = multiprocessing.Pool(
            processes=self.request_num_cpus
        )  # Set num_processes to the desired number of parallel processes

        # Map the experiment_ids to the download_experiment function in parallel
        results = pool.map(self.download_experiment, self.experiment_ids)

        # Close the pool to release resources
        pool.close()
        pool.join()

        return self.metadata

    def download_experiment(self, experiment_id: int):
        """
        Downloads metadata of 1 experiment (whether a screen or a project).

        Arguments
        ---------
        experiment_id (int): ID of the experiment to download from IDR database

        Returns
        -------
        metadata (dict)
        """

        # Browse API for 'screen' and 'project' references (can have same id sometimes)
        for experiment_type in self.experiment_types:
            url = f"https://idr.openmicroscopy.org/webclient/api/annotations/?type=map&{experiment_type}={experiment_id}"

            # If cannot find experiment_id, skip
            if self.session.get(url).json()["annotations"] != []:
                # Instanciate EXPERIMENT IMAGES counter
                total_nb_images_in_experiment = 0

                for a in self.session.get(url).json()["annotations"]:
                    # STOP IF BATCH SIZE IS REACHED
                    if self.total_nb_images >= self.batch_size:
                        tqdm.write(
                            f"Batch size of {self.batch_size} reached. Stopping."
                        )
                        self.metadata["total_experiments"] = (
                            self.screens_counter + self.projects_counter
                        )
                        self.metadata["total_images"] = self.total_nb_images
                        self.dump_json()
                        return self.metadata

                    # Check if screen is exploitable, or if it is a symlink for other screens (e.g. 'screen' 1502)
                    if a["values"][0][0] == "Sample Type":
                        # Dump in json only sample type screens that correspond to 'type' (not interested in retrieving all screens)
                        if self.data_type != "all":
                            if a["values"][0][1] == self.data_type:
                                tqdm.write(
                                    f"Retrieving metadata for {experiment_type} n°{experiment_id}"
                                )
                                self.retrieve_experiment(
                                    a=a,
                                    experiment_type=experiment_type,
                                    experiment_id=experiment_id,
                                    total_nb_images_in_experiment=total_nb_images_in_experiment,
                                )

                        elif self.data_type == "all":
                            tqdm.write(
                                f"Retrieving metadata for {experiment_type} n°{experiment_id}"
                            )
                            self.retrieve_experiment(
                                a=a,
                                experiment_type=experiment_type,
                                experiment_id=experiment_id,
                                total_nb_images_in_experiment=total_nb_images_in_experiment,
                            )

                        else:
                            raise ValueError(
                                "Data type must be 'cell', 'tissue' or 'all'"
                            )

                        if experiment_type == "screen":
                            self.screens_counter += 1

                        elif experiment_type == "project":
                            self.projects_counter += 1

                        self.metadata["total_experiments"] = (
                            self.screens_counter + self.projects_counter
                        )
                        self.metadata["total_images"] = self.total_nb_images

                        # Dump in json
                        self.dump_json(experiment_id=experiment_id)

                    else:
                        continue

            else:
                continue

        assert (
            bool(self.metadata["screens"]) or bool(self.metadata["projects"])
        ), f"Experiment n° {experiment_id} does not exist in {self.data_type} IDR database."

        return self.metadata

    def get_experiments_ids(self):
        """
        Retrieve all the existing experiments IDs and input them in a .csv file (for Condor parallelization).
        """

        # Launch session to connect to API
        self.launch_session()

        experiment_ids = []

        # The highest experiment_id number in IDR (CELL) less than 3500 (as of 10/05/23)
        for experiment_id in tqdm(range(0, 3350)):
            # -------------------------------------------------------------------------------------------------------- #
            # --------------------------------------------- EXPERIMENT Level ----------------------------------------- #
            # -------------------------------------------------------------------------------------------------------- #

            # Browse API for 'screen' and 'project' references (can have same id sometimes)
            for experiment_type in self.experiment_types:
                url = f"https://idr.openmicroscopy.org/webclient/api/annotations/?type=map&{experiment_type}={experiment_id}"

                # If cannot find experiment_id, skip
                if self.session.get(url).json()["annotations"] != []:
                    for a in self.session.get(url).json()["annotations"]:
                        # Check if screen is exploitable, or if it is a symlink for other screens (e.g. 'screen' 1502)
                        if a["values"][0][0] == "Sample Type":
                            # Dump in json only sample type screens that correspond to 'type' (not interested in retrieving all screens)
                            if self.data_type != "all":
                                if a["values"][0][1] == self.data_type:
                                    experiment_ids.append(experiment_id)

        # Create a DataFrame with the experiment IDs
        df = pd.DataFrame({"experiment_id": experiment_ids})

        # Remove duplicates
        df = df.drop_duplicates()

        # Save the DataFrame to a .csv file
        df.to_csv(f"experiment_ids_{self.data_type}.csv", index=False)

    def launch_session(self):
        """
        Launch a session on IDR website.

        Returns
        -------
        session (requests.Session): session on IDR website
        """

        INDEX_PAGE = f"{self.base_url}/webclient/?experimenter=-1"

        # create http session
        with requests.Session() as session:
            request = requests.Request("GET", INDEX_PAGE)
            prepped = session.prepare_request(request)
            response = session.send(prepped)
            if response.status_code != 200:
                response.raise_for_status()

        self.session = session

    def dump_json(self, experiment_id):
        """
        Dump metadata in json file.
        """

        # Dump in json
        with open(f"{self.filename}_{experiment_id}.json", "w") as f:
            json.dump(self.metadata, f)

    def retrieve_experiment(
        self,
        a: dict,
        experiment_type: str,
        experiment_id: int,
        total_nb_images_in_experiment: int,
    ):
        """
        Retrieve metadata for a given experiment.

        Parameters
        ----------
        session (requests.Session): session on IDR website
        metadata (dict): metadata to dump
        a (dict): annotation
        experiment_type (str): type of experiment (screen or project)
        experiment_id (int): experiment id
        screens_counter (int): number of screens
        projects_counter (int): number of projects
        total_nb_images (int): total number of images
        nb_images_experiment (int): number of images in experiment
        image_metadata (bool): whether to download image metadata or not

        Returns
        -------
        metadata (dict): metadata to dump
        screens_counter (int): number of screens
        """

        # --------------------------------------- SCREEN Case ------------------------------------------- #
        if experiment_type == "screen":
            # ----------------------------- Retrieve EXPERIMENT MAP ANNOTATIONS ----------------------------- #
            self.retrieve_experiment_annotations(
                a=a, experiment_type=experiment_type, experiment_id=experiment_id
            )

            # ----------------------------- Retrieve PLATES in given SCREEN ----------------------------- #
            plates_counter, total_nb_images_in_experiment = self.retrieve_plates(
                experiment_id=experiment_id,
                total_nb_images_in_experiment=total_nb_images_in_experiment,
            )

            # ---------------------- Check if plates in screen; if not, delete screen ------------------- #
            if plates_counter == 0 and total_nb_images_in_experiment == 0:
                del self.metadata["screens"][self.screens_counter]

            else:
                self.metadata["screens"][self.screens_counter]["total_plates"] = (
                    plates_counter
                )
                self.metadata["screens"][self.screens_counter][
                    "total_images_in_screen"
                ] = total_nb_images_in_experiment

            return 0

        # --------------------------------------- PROJECT Case ------------------------------------------- #
        elif experiment_type == "project":
            # ----------------------------- Retrieve EXPERIMENT MAP ANNOTATIONS ----------------------------- #
            self.retrieve_experiment_annotations(
                a=a, experiment_type=experiment_type, experiment_id=experiment_id
            )

            # ----------------------------- Retrieve DATASETS in given PROJECT --------------------------- #
            datasets_counter, total_nb_images_in_experiment = self.retrieve_datasets(
                experiment_id=experiment_id,
                total_nb_images_in_experiment=total_nb_images_in_experiment,
            )

            # ---------------------- Check if images in dataset; if not, delete screen ------------------- #
            if datasets_counter == 0 and total_nb_images_in_experiment == 0:
                del self.metadata["projects"][self.projects_counter]

            else:
                self.metadata["projects"][self.projects_counter]["total_datasets"] = (
                    datasets_counter
                )
                self.metadata["projects"][self.projects_counter][
                    "total_images_in_project"
                ] = total_nb_images_in_experiment

            return 0

    def retrieve_experiment_annotations(
        self, a: dict, experiment_type: str, experiment_id: int
    ):
        """
        Retrieve experiment annotations.
        """
        experiments_type = experiment_type + "s"
        counter = (
            self.screens_counter
            if experiment_type == "screen"
            else self.projects_counter
        )

        self.metadata[experiments_type][counter] = {}
        self.metadata[experiments_type][counter]["experiment_id"] = experiment_id
        self.metadata[experiments_type][counter]["experiment_map_annotation"] = {}

        for v in a["values"]:
            key = v[0]
            value = v[1]
            # Write in dict
            self.metadata[experiments_type][counter]["experiment_map_annotation"][
                key
            ] = value

        return 0

    def retrieve_plates(self, experiment_id: int, total_nb_images_in_experiment: int):
        """
        Retrieve plates in given screen.

        Parameters
        ----------
        metadata (dict): metadata dict
        experiment_id (int): screen id
        screens_counter (int): screen counter
        session (requests.Session): session on IDR website
        total_nb_images (int): number of images in dataset
        nb_images_experiment (int): number of images in screen
        nb_images_plate (int): number of images in plate
        image_metadata (bool): whether to download image metadata or not

        Returns
        -------
        metadata (dict): metadata dict
        """

        PLATES_URL = f"{self.base_url}/webclient/api/plates/?id={experiment_id}"
        self.metadata["screens"][self.screens_counter]["plates"] = {}

        # Intanciate PLATES counter
        plates_counter = 0

        if self.session.get(PLATES_URL).json()["plates"] != []:
            tqdm.write(f"Retrieving plates of screen n° {experiment_id} ...")
            for plate in tqdm(
                self.session.get(PLATES_URL).json()["plates"], file=sys.stdout
            ):
                # Instanciate PLATE IMAGES counter
                nb_images_plate = 0

                # -------------------------------------------------------------------------------------------------------- #
                # --------------------------------------------- PLATE Level --------------------------------------------- #
                # -------------------------------------------------------------------------------------------------------- #

                plate_id = plate["id"]
                plate_name = plate["name"]
                plate_childCount = plate["childCount"]
                self.metadata["screens"][self.screens_counter]["plates"][
                    plates_counter
                ] = {}
                self.metadata["screens"][self.screens_counter]["plates"][
                    plates_counter
                ]["plate_id"] = plate_id
                self.metadata["screens"][self.screens_counter]["plates"][
                    plates_counter
                ]["plate_name"] = plate_name
                self.metadata["screens"][self.screens_counter]["plates"][
                    plates_counter
                ]["plate_childCount"] = plate_childCount
                self.metadata["screens"][self.screens_counter]["plates"][
                    plates_counter
                ]["wells"] = {}

                # Instanciate FIELDS counter
                field_number = 0

                image_sizes = {"x": 1080, "y": 1080}

                while image_sizes != []:
                    # -------------------------------------------------------------------------------------------------------- #
                    # --------------------------------------------- WELL Level --------------------------------------------- #
                    # -------------------------------------------------------------------------------------------------------- #

                    WELLS_IMAGES_URL = "{base}/webgateway/plate/{plate_id}/{field}/"
                    qs = {
                        "base": self.base_url,
                        "plate_id": plate_id,
                        "field": field_number,
                    }
                    url = WELLS_IMAGES_URL.format(**qs)
                    grid = self.session.get(url).json()
                    for row in grid["grid"]:
                        for cell in row:
                            if cell is not None:
                                # Check if well_id already in dict
                                if (
                                    cell["wellId"]
                                    not in self.metadata["screens"][
                                        self.screens_counter
                                    ]["plates"][plates_counter]["wells"]
                                ):
                                    self.metadata["screens"][self.screens_counter][
                                        "plates"
                                    ][plates_counter]["wells"][cell["wellId"]] = {}
                                    self.metadata["screens"][self.screens_counter][
                                        "plates"
                                    ][plates_counter]["wells"][cell["wellId"]][
                                        "well_id"
                                    ] = cell["wellId"]
                                    self.metadata["screens"][self.screens_counter][
                                        "plates"
                                    ][plates_counter]["wells"][cell["wellId"]][
                                        "fields"
                                    ] = {}

                                # -------------------------------------------------------------------------------------------------------- #
                                # --------------------------------------------- IMAGE Level --------------------------------------------- #
                                # -------------------------------------------------------------------------------------------------------- #

                                self.metadata["screens"][self.screens_counter][
                                    "plates"
                                ][plates_counter]["wells"][cell["wellId"]]["fields"][
                                    field_number
                                ] = {}
                                self.metadata["screens"][self.screens_counter][
                                    "plates"
                                ][plates_counter]["wells"][cell["wellId"]]["fields"][
                                    field_number
                                ]["image_id"] = cell["id"]
                                self.metadata["screens"][self.screens_counter][
                                    "plates"
                                ][plates_counter]["wells"][cell["wellId"]]["fields"][
                                    field_number
                                ]["field"] = field_number

                                self.total_nb_images += 1
                                total_nb_images_in_experiment += 1
                                nb_images_plate += 1

                                if self.image_metadata:
                                    # Retrieve Metadata of given image (OPTIONAL)
                                    self.metadata["screens"][self.screens_counter][
                                        "plates"
                                    ][plates_counter]["wells"][cell["wellId"]][
                                        "fields"
                                    ][field_number]["image_metadata"] = {}

                                    MAP_URL = f"{self.base_url}/webclient/api/annotations/?type=map&image={cell['id']}"

                                    for a in self.session.get(MAP_URL).json()[
                                        "annotations"
                                    ]:
                                        for v in a["values"]:
                                            key = v[0]
                                            value = v[1]
                                            self.metadata["screens"][
                                                self.screens_counter
                                            ]["plates"][plates_counter]["wells"][
                                                cell["wellId"]
                                            ]["fields"][field_number]["image_metadata"][
                                                str(key)
                                            ] = value
                                            # Adds num_channels
                                            if key == "Channels":
                                                self.metadata["screens"][
                                                    self.screens_counter
                                                ]["plates"][plates_counter]["wells"][
                                                    cell["wellId"]
                                                ]["fields"][field_number][
                                                    "image_metadata"
                                                ]["num_channels"] = len(
                                                    value.strip().split(";")
                                                )

                                # BREAK IF BATCH SIZE IS REACHED
                                if self.total_nb_images >= self.batch_size:
                                    self.metadata["screens"][self.screens_counter][
                                        "plates"
                                    ][plates_counter][
                                        "total_images_in_plate"
                                    ] = nb_images_plate
                                    return plates_counter, total_nb_images_in_experiment

                    image_sizes = grid["image_sizes"]
                    field_number += 1

                self.metadata["screens"][self.screens_counter]["plates"][
                    plates_counter
                ]["total_images_in_plate"] = nb_images_plate
                plates_counter += 1

            return plates_counter, total_nb_images_in_experiment

        else:
            return plates_counter, total_nb_images_in_experiment

    def retrieve_datasets(self, experiment_id: int, total_nb_images_in_experiment: int):
        """
        Retrieve datasets in given project.

        Parameters
        ----------
        experiment_id (int): screen id
        total_nb_images_in_experiment (int): number of images in project


        Returns
        -------
        int : 0 if ok
        """

        DATASETS_URL = (
            f"{self.base_url}/webclient/api/datasets/?id={experiment_id}&limit=30000"
        )

        self.metadata["projects"][self.projects_counter]["datasets"] = {}

        # Intanciate PLATES counter
        datasets_counter = 0

        if self.session.get(DATASETS_URL).json()["datasets"] != []:
            tqdm.write(f"Retrieving datasets of project n° {experiment_id} ...")
            for dataset in tqdm(
                self.session.get(DATASETS_URL).json()["datasets"], file=sys.stdout
            ):
                # Instanciate DATASET IMAGES counter
                nb_images_dataset = 0

                # -------------------------------------------------------------------------------------------------------- #
                # --------------------------------------------- DATASET Level --------------------------------------------- #
                # -------------------------------------------------------------------------------------------------------- #

                dataset_id = dataset["id"]
                dataset_name = dataset["name"]
                dataset_childCount = dataset["childCount"]
                self.metadata["projects"][self.projects_counter]["datasets"][
                    datasets_counter
                ] = {}
                self.metadata["projects"][self.projects_counter]["datasets"][
                    datasets_counter
                ]["dataset_id"] = dataset_id
                self.metadata["projects"][self.projects_counter]["datasets"][
                    datasets_counter
                ]["dataset_name"] = dataset_name
                self.metadata["projects"][self.projects_counter]["datasets"][
                    datasets_counter
                ]["dataset_childCount"] = dataset_childCount
                self.metadata["projects"][self.projects_counter]["datasets"][
                    datasets_counter
                ]["images"] = {}

                # Instanciate IMAGES counter
                image_number = 0

                IMAGES_URL = f"{self.base_url}/api/v0/m/datasets/{dataset_id}/images/"
                qs = {"base": self.base_url, "dataset_id": dataset_id}
                url = IMAGES_URL.format(**qs)
                data = self.session.get(url).json()

                for row in data["data"]:
                    self.metadata["projects"][self.projects_counter]["datasets"][
                        datasets_counter
                    ]["images"][image_number] = {}
                    self.metadata["projects"][self.projects_counter]["datasets"][
                        datasets_counter
                    ]["images"][image_number]["image_id"] = row["@id"]

                    if self.image_metadata:
                        # Retrieve Metadata of given image (OPTIONAL)
                        self.metadata["projects"][self.projects_counter]["datasets"][
                            datasets_counter
                        ]["images"][image_number]["image_metadata"] = {}

                        MAP_URL = f"{self.base_url}/webclient/api/annotations/?type=map&image={row['@id']}"

                        for a in self.session.get(MAP_URL).json()["annotations"]:
                            for v in a["values"]:
                                key = v[0]
                                value = v[1]
                                self.metadata["projects"][self.projects_counter][
                                    "datasets"
                                ][datasets_counter]["images"][image_number][
                                    "image_metadata"
                                ][str(key)] = value
                                # Adds num_channels
                                if key == "Channels":
                                    self.metadata["projects"][self.projects_counter][
                                        "datasets"
                                    ][datasets_counter]["images"][image_number][
                                        "image_metadata"
                                    ]["num_channels"] = len(value.strip().split(";"))

                    self.total_nb_images += 1
                    total_nb_images_in_experiment += 1
                    nb_images_dataset += 1
                    image_number += 1

                    # BREAK IF BATCH SIZE IS REACHED
                    if self.total_nb_images >= self.batch_size:
                        self.metadata["projects"][self.projects_counter]["datasets"][
                            datasets_counter
                        ]["total_images_in_dataset"] = nb_images_dataset
                        datasets_counter += 1
                        return datasets_counter, total_nb_images_in_experiment

                self.metadata["projects"][self.projects_counter]["datasets"][
                    datasets_counter
                ]["total_images_in_dataset"] = nb_images_dataset
                datasets_counter += 1

            return datasets_counter, total_nb_images_in_experiment

        else:
            return datasets_counter, total_nb_images_in_experiment


def main(args):
    parser = argparse.ArgumentParser()

    # Argument parser

    parser.add_argument(
        "--filename",
        type=str,
        default="metadata.json",
        help="Name of the json file to save the metadata",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=2048,
        help="Batch size of metadata file.",
    )

    parser.add_argument(
        "--experiment_ids",
        type=int,
        default=1,
        help="List or int of experiment_id(s) to download metadata for.",
    )

    parser.add_argument(
        "--image_metadata",
        action="store_true",
        help="Whether to download image metadata or not",
    )

    parser.add_argument(
        "--data_type",
        type=str,
        default="cell",
        help="Type of screen to download",
    )

    parser.add_argument(
        "--request_num_cpus",
        type=int,
        default=1,
        help="Number of CPUs to use for multiprocessing.",
    )

    args = parser.parse_args()

    downloader = MetaDataDownloader(
        experiment_ids=args.experiment_ids,
        request_num_cpus=args.request_num_cpus,
        filename=args.filename,
        batch_size=args.batch_size,
        image_metadata=args.image_metadata,
        data_type=args.data_type,
    )
    metadata = downloader.download()


if __name__ == "__main__":
    main(sys.argv[1:])
