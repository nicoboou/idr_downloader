# Standard imports
import os
import sys
import argparse
import random

# Third party libraries
import numpy as np
import matplotlib.pyplot as plt
import ijson
from tqdm import tqdm
from PIL import Image
import cv2
import time

import omero
from idr import connections

dir_path = os.getcwd()
sys.path.insert(0, dir_path)

print(f"Current directory: {dir_path}")

from src.modules.datadownloaders.base_downloader import BaseDownloader


class DataDownloader(BaseDownloader):
    """
    Class to download data from IDR.

    Attributes
    ----------
    metadata_filepath (str): Path to the metadata.json file

    """

    def __init__(
        self,
        metadata_filepath: str,
        file_format: str,
        save_path: str,
        cuda_devices: list = [0],
        use_multi_cpus: bool = False,
        use_multi_threads: bool = False,
        max_num_cpus: int = 1,
        max_num_threads: int = 1,
    ):
        super().__init__(
            cuda_devices=cuda_devices,
            use_multi_cpus=use_multi_cpus,
            use_multi_threads=use_multi_threads,
            max_num_cpus=max_num_cpus,
            max_num_threads=max_num_threads,
        )
        self.metadata_filepath = metadata_filepath
        self.file_format = file_format
        self.save_path = save_path

    def download_single_image(
        self,
        experiment_id: int,
        image_id: int,
        resize_size: int,
        crop_size: int,
        normalize: bool = True,
        stack_channels: bool = True,
    ):
        (
            image_numpy,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = self.retrieve_image(
            image_id=image_id,
            normalize=normalize,
            resize_size=resize_size,
            crop_size=crop_size,
        )

        # Save images directly
        print(self.file_format)
        if self.file_format == "png":
            filename = self.save_path + str(experiment_id)
            print(f"Saving image {image_id} on the fly...")
            DataDownloader.save_to_png(
                images=[image_numpy],
                labels=[image_id],
                filename=filename,
                stack_channels=stack_channels,
            )
            print(f"Image {image_id} saved.")

        elif self.file_format == "npz":
            # Save tiny dataset to npz file
            filename = self.save_path + str(experiment_id) + ".npz"
            print(f"Saving image {image_id} on the fly...")
            DataDownloader.save_to_npz(
                images=[image_numpy],
                labels=[image_id],
                filename=filename,
            )
            print(f"Image {image_id} saved.")

    def download(
        self,
        num_images_requested: int,
        normalize: bool = True,
        resize_size: int = 0,
        crop_size: int = 0,
        stack_channels: bool = False,
    ) -> list:
        """
        Download images from IDR

        Parameters
        ----------
        num_images_requested (int): Number of images to download
        normalize (bool): Normalize images or not
        resize_size (int): Resize images to this size
        crop_size (int): Crop images to this size
        stack_channels (bool): Stack channels or save channel per channel

        Returns
        -------
        full_images_numpy (list): returns a LIST of numpy arrays
        """

        print("Start downloading images per experiment...")

        # Get image ids
        image_ids = self.retrieve_image_ids(metadata_file=self.metadata_filepath)

        # Randomly select image ids
        random_image_ids_selected = self.select_random_image_ids(
            image_ids=image_ids, num_image_ids_requested=num_images_requested
        )

        try:
            for experiment_id in tqdm(random_image_ids_selected.keys(), file=sys.stdout, desc="Experiments"):
                if bool(random_image_ids_selected[experiment_id]):
                    image_ids_list = random_image_ids_selected[experiment_id]
                    tasks = [
                        (
                            self.download_single_image,
                            (experiment_id, image_id, resize_size, crop_size, normalize, stack_channels),
                        )
                        for image_id in image_ids_list
                    ]
                    self.execute_tasks_multiprocesses(tasks)
            return 0
        except Exception as e:
            print(f"Error downloading images: {e}")
            return 1

    # =================================== IMAGE IDS SELECTION FUNCTIONS =================================== #

    @staticmethod
    def get_image_ids(
        metadata,
        num_projects,
        num_screens,
        num_images_per_experiment,
        randomize=True,
        seed=42,
    ):
        """
        Get random image ids from metadata.

        Arguments
        ---------
        metadata (json): JSON file containing metadata
        num_projects (int): Number of projects to select
        num_screens (int): Number of screens to select

        Returns
        -------
        image_ids (list): List of image ids
        """

        # Set seed
        random.seed(seed)

        image_ids = []
        projects = metadata.get("projects", {})
        screens = metadata.get("screens", {})
        project_ids = list(projects.keys())
        screen_ids = list(screens.keys())

        # Randomly select projects
        if len(project_ids) > 0:
            random_project_ids = random.sample(project_ids, min(num_projects, len(project_ids)))

            for project_id in random_project_ids:
                datasets = projects[project_id].get("datasets", {})
                for dataset in datasets.values():
                    images = dataset.get("images", {})
                    image_ids.extend(image["image_id"] for image in images.values())

        # Randomly select screens
        if len(screen_ids) > 0:
            random_screen_ids = random.sample(screen_ids, min(num_screens, len(screen_ids)))

            for screen_id in random_screen_ids:
                plates = screens[screen_id].get("plates", {})
                for plate in plates.values():
                    wells = plate.get("wells", {})
                    for well in wells.values():
                        fields = well.get("fields", {})
                        for field in fields.values():
                            image_ids.append(field["image_id"])
        if randomize:
            random.shuffle(image_ids)

        if num_images_per_experiment:
            return image_ids[: num_images_per_experiment * (num_projects + num_screens)]

        else:
            return image_ids

    @staticmethod
    def select_random_image_ids(image_ids: dict, num_image_ids_requested: int) -> dict:
        """
        Select random images from a list of image ids.

        Arguments
        ---------
        image_ids_list (dict): dict of image ids per experiment
        num_image_ids_requested (int): Number of image ids to select

        Returns
        -------
        random_image_ids (dict): List of randomly selected image ids
        """

        for experiment_id in image_ids.keys():
            num_images_available = len(image_ids[experiment_id])
            if num_image_ids_requested <= num_images_available:
                image_ids[experiment_id] = random.sample(image_ids[experiment_id], num_image_ids_requested)

            else:
                print(
                    f"Requested {num_image_ids_requested} images for {experiment_id}, but only {num_images_available} images available. \n Not downloading {experiment_id}"
                )
                image_ids[experiment_id] = {}

        return image_ids

    @staticmethod
    def retrieve_image_ids(metadata_file: str) -> dict:
        """
        Retrieve ALL image ids from metadata file.

        Arguments
        ---------
        metadata_file (str): Path to the metadata file

        Returns
        -------
        image_ids (dict): Dictionary of image ids
        """

        image_ids = {}

        with open(metadata_file, "rb") as f:
            parser = ijson.parse(f)

            current_experiment_type = None
            current_experiment_id = None

            for prefix, event, value in parser:
                if event == "map_key":
                    # Store the current key to determine the data to extract
                    current_key = value

                    if current_key == "screens":
                        # Start of screens section, update the current experiment type
                        current_experiment_type = "screen"

                    elif current_key == "projects":
                        # Start of projects section, update the current experiment type
                        current_experiment_type = "project"

                    elif current_key == "experiment_id":
                        # Store the experiment ID for the current experiment type
                        _, _, current_experiment_id = next(parser)

                    elif current_key == "image_id":
                        # Store the image ID under the current experiment type and ID
                        _, _, image_id = next(parser)

                        if current_experiment_type and current_experiment_id:
                            image_ids.setdefault(
                                current_experiment_type + "_" + str(current_experiment_id),
                                [],
                            ).append(image_id)

        return image_ids

    # =================================== IMAGE DOWNLOAD FUNCTIONS =================================== #

    def retrieve_image(
        self,
        image_id: int,
        normalize: bool = True,
        resize_size: int = 0,
        crop_size: int = 0,
    ):
        """
        Get image from IDR

        Parameters
        ----------
        image_id (int): IDR image id
        normalize (bool): Normalize image or not

        Returns
        -------
        image_numpy (np.array): Numpy array of image
        """
        # Connect to IDR
        conn = connections.connection("idr.openmicroscopy.org", verbose=2)

        try:
            # Sleep to avoid bangering the API
            random_time_sleep = random.uniform(1.2, 12.7)
            time.sleep(random_time_sleep)

            image = conn.getObject("Image", image_id)
            # print(f'Image retrieved')

            # Query pixelsWrapper
            pixels = image.getPrimaryPixels()
            # print('Primary pixels retrieved')

            # Get channelsWrapper
            channels = image.getChannels()
            # print('Channels retrieved')

            # Get number of Z-stacks
            size_z = image.getSizeZ()
            print(f"size_z:{size_z}")

            # Get number of channels
            size_c = image.getSizeC()

            # Get number of timepoints
            size_t = image.getSizeT()
            print(f"size_t: {size_t}")

            # Get image dimensions
            size_y = image.getSizeY()
            size_x = image.getSizeX()

            # N.B. : ZCT order, here we have 3 usecases:
            # - 2D images: size_z = 1
            # - 3D images: size_z > 1
            # - 4D images: size_t > 1
            # => in any case, we want to get the first timepoint and the floor(len(size_z)/2) z-stack (most precise focal plane)

            z_stack = int(np.floor(size_z / 2))
            image_numpy = self.stack_channels(
                pixels=pixels,
                size_x=size_x,
                size_y=size_y,
                size_c=size_c,
                z_stack=z_stack,
                normalize=normalize,
                resize_size=resize_size,
                crop_size=crop_size,
            )

            # Close connection
            # print('Closing connection...')
            conn.close()
            # print('Connection closed !')

            return image_numpy, channels, size_z, size_c, size_t, size_y, size_x

        except Exception as e:
            print(f"ERROR: {e}")
            print("Closing connection to the BlitzGateway...")
            conn.close()

    # =================================== IMAGE PROCESSING FUNCTIONS =================================== #

    @staticmethod
    def stack_channels(
        pixels,
        size_x: int,
        size_y: int,
        size_c: int,
        z_stack: int,
        normalize: bool = True,
        resize_size: int = 0,
        crop_size: int = 0,
    ):
        """
        Stack channels into a single numpy array
        """
        image_stack = []

        # Assert that resize and crop are not both requested
        assert not (resize_size != 0 and crop_size != 0), "Cannot resize and crop image at the same time"

        for idx in range(0, size_c):
            # CROP image if requested
            if crop_size != 0:
                # Compute the tile to get from the center of the image
                x = (size_x - crop_size) // 2  #
                y = (size_y - crop_size) // 2
                tile = (x, y, crop_size, crop_size)
                print(f"Tile: {tile}")

                # get the Tile
                image_plane = pixels.getTile(z_stack, idx, 0, tile)

            # RESIZE image if requested
            else:
                image_plane = pixels.getPlane(z_stack, idx, 0)
                if resize_size != 0:
                    image_plane = DataDownloader.resize_channel(image_plane, resize_size)

            # Normalize image if requested
            if normalize:
                image_plane = DataDownloader.normalize_channel(image_plane)

            image_stack.append(image_plane)

        return np.dstack(image_stack)

    @staticmethod
    def normalize_channel(channel: np.array):
        """
        Normalize image with the following steps:
        - Subtract 1st percentile value
        - Divide by the difference between 99.99th percentile and 1st percentile
        - Clip values between 0 and 1
        - Multiply by 255
        - Convert to uint8

        Parameters
        ----------
        channel (np.array): Numpy array of image channel

        Returns
        -------
        uint8_channel (np.array): Normalized numpy array of image channel
        """

        min_percentile = np.percentile(channel, 1)
        max_percentile = np.percentile(channel, 99.99)
        channel_range = max_percentile - min_percentile

        normalized_channel = (channel - min_percentile) / channel_range
        normalized_channel = np.clip(normalized_channel, 0, 1)
        scaled_channel = normalized_channel * 255
        uint8_channel = scaled_channel.astype(np.uint8)

        return uint8_channel

    @staticmethod
    def resize_channel(channel: np.array, resize_size: int) -> np.array:
        """
        Resize image to target width and height using two different methods:
        - INTER_AREA: Interpolate if we shrink image
        - INTER_CUBIC: Interpolate if we enlarge image

        Parameters
        ----------
        channel (np.array): Numpy array of channel
        resize_size (int): Desired width and height of the resized image

        Returns
        -------
        resized_channel (np.array): Resized numpy array of channel
        """

        # Get channel dimensions
        size_y = channel.shape[0]
        size_x = channel.shape[1]

        # Calculate the aspect ratio of the original channel
        aspect_ratio = size_x / size_y

        # Determine the actual width and height based on the target size and aspect ratio
        if resize_size is not None:
            target_width = resize_size
            target_height = resize_size
        else:
            target_width = size_x
            target_height = size_y

        #  Enlarge the channel if necessary
        if target_width < size_x or target_height < size_y:
            resized_channel = cv2.resize(
                channel, (target_width, target_height), interpolation=cv2.INTER_CUBIC
            )

        # Shrinking the channel
        elif target_width >= size_x or target_height >= size_y:
            resized_channel = cv2.resize(channel, (target_width, target_height), interpolation=cv2.INTER_AREA)

        else:
            resized_channel = channel

        return resized_channel

    @staticmethod
    def compute_channel_entropy(channel: np.array) -> float:
        # Flatten the channel into a 1D array
        flattened_channel = channel.flatten()

        # Compute the histogram of pixel values
        histogram, _ = np.histogram(flattened_channel, bins=256, range=(0, 255))

        # Normalize the histogram to get probabilities
        probabilities = histogram / np.sum(histogram)

        # Filter out probabilities equal to zero
        nonzero_probabilities = probabilities[probabilities != 0]

        # Compute the entropy using the Shannon entropy formula
        entropy = -np.sum(nonzero_probabilities * np.log2(nonzero_probabilities))

        return entropy

    @staticmethod
    def compute_image_entropy(image: np.array) -> float:
        """
        Compute the entropy of an image

        Parameters
        ----------
        image (np.array): Numpy array of image

        Returns
        -------
        entropy (float): Entropy of the image
        """
        # Initialize an empty list to hold the channel entropies
        channel_entropies = []

        # Compute the entropy of each channel
        for channel_idx in range(image.shape[2]):
            channel_entropy = DataDownloader.compute_channel_entropy(image[:, :, channel_idx])
            channel_entropies.append(channel_entropy)

        # Compute the average entropy across all channels
        entropy = np.mean(channel_entropies)

        return entropy

    # =================================== IMAGE PLOT FUNCTIONS =================================== #

    @staticmethod
    def plot_image_channels(image: omero.gateway._ImageWrapper):
        """
        Plot image channels

        Parameters
        ----------
        image (omero.gateway._ImageWrapper): Image object from IDR
        """
        # Query pixelsWrapper
        pixels = image.getPrimaryPixels()

        # Get number of channels
        size_c = image.getSizeC()

        plt.figure(figsize=(25, 20))
        for idx in range(0, size_c):
            plt.subplot(1, 5, idx + 1)
            image_plane = pixels.getPlane(0, idx, 0)
            plt.imshow(image_plane, cmap="gray")
            plt.axis("off")
            plt.title("Channel " + str(idx))

        plt.show()

    # =================================== IMAGE SAVING FUNCTIONS =================================== #

    @staticmethod
    def save_to_npz(images: list, labels, filename: str):
        # Save image stack
        print(f"Saving image stack to {filename}...")
        if labels:
            # Save each array as a separate file in the .npz archive
            np.savez(filename, **{str(label): array for label, array in zip(labels, images)})
        else:
            np.savez(filename, *images)
        print("Image stack saved !")

    @staticmethod
    def load_npz(file_path: str):
        data = np.load(file_path)
        print(f"Data file loaded... \n Shape of numpy array: {data['images'].shape}")
        return data

    @staticmethod
    def save_to_png(images: list, labels: list, filename: str, stack_channels=False):
        if stack_channels:
            for idx, image_numpy in enumerate(images):
                image_array = image_numpy.astype(np.uint8)
                image = Image.fromarray(image_array)
                image.save(f"{filename}_{labels[idx]}.png")
        else:
            for idx, image_numpy in enumerate(images):
                for channel_idx in range(image_numpy.shape[2]):
                    channel_data = image_numpy[:, :, channel_idx].astype(np.uint8)
                    channel_image = Image.fromarray(channel_data, mode="L")
                    channel_image.save(f"{filename}_{labels[idx]}_{channel_idx}.png")

    @staticmethod
    def save_to_jpeg(images: list, labels: list, filename: str):
        for idx, image_numpy in enumerate(images):
            # Iterate over the channels
            for channel_idx in range(image_numpy.shape[2]):
                # Extract the channel data
                channel_data = image_numpy[:, :, channel_idx]

                # Create a PIL Image object from the channel data
                channel_image = Image.fromarray(channel_data, mode="L")

                # Save the channel image as a PNG file
                channel_image.save(f"{filename}_{labels[idx]}_{channel_idx}.jpeg")

    @staticmethod
    def save_to_tiff(images: list, labels: list, filename: str):
        for idx, image_numpy in enumerate(images):
            channels = np.array([])

            # Iterate over the channels
            for channel_idx in range(image_numpy.shape[2]):
                # Extract the channel data
                np.append(channels, image_numpy[:, :, channel_idx])

                # Save the channel image as a PNG file
            image = Image.fromarray(channels)
            image.save(f"{filename}_{labels[idx]}.tiff")

    @staticmethod
    def train_test_split(data: np.array, test_size_perc: float, shuffle=True, random_seed=42):
        """
        Split data into train and test sets.

        Parameters
        ----------
        data (np.array): Numpy array of data
        test_split (float): Percentage of data to use for test set

        Returns
        -------
        train_data (np.array): Numpy array of training data
        test_data (np.array): Numpy array of test data
        """

        # Take length of data
        n = len(data)

        # Compute test size
        test_size = int(n * test_size_perc)

        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)

        if shuffle:
            # Shuffle array on axis 0
            np.random.shuffle(data)

            # Create indices for shuffled data
            indices = np.arange(n)
            np.random.shuffle(indices)

        else:
            indices = np.arange(n)

        # Split indices
        test_indices = indices[:test_size]
        train_indices = indices[test_size:]

        # Split data
        train_data = data[train_indices]
        test_data = data[test_indices]

        return train_data, test_data

    @staticmethod
    def split_data(data: np.array, test_size: float, val_size: float, random_seed=42):
        """
        Split data into train, validation and test sets.

        Parameters
        ----------
        data (np.array): Numpy array of data
        test_size (float): Percentage of data to use for test set
        val_size (float): Percentage of data to use for validation set

        Returns
        -------
        train_data (np.array): Numpy array of training data
        val_data (np.array): Numpy array of validation data
        test_data (np.array): Numpy array of test data
        """

        print(
            f"Splitting data into train, validation and test sets... \n Test size: {test_size} \n Validation size: {val_size}"
        )
        # Split the data into TRAIN and TEST sets
        train_data, test_data = DataDownloader.train_test_split(
            data, test_size_perc=test_size, shuffle=True, random_seed=random_seed
        )

        # Split the train set into TRAIN and VAL sets
        train_data, val_data = DataDownloader.train_test_split(
            train_data, test_size_perc=val_size, shuffle=True, random_seed=random_seed
        )

        return train_data, val_data, test_data

    @staticmethod
    def retrieve_3D_image(image):
        """
        Retrieve 3D image from IDR
        # adapted from https://github.com/ome/omero-guide-ilastik/blob/e9df8014515a8dbbfd87623a24564044d05d2224/notebooks/pixel_classification.ipynb

        Parameters
        ----------
        image (omero.gateway._ImageWrapper): 3D image object from IDR

        Returns
        -------
        image_numpy (np.array): Numpy array of image
        """

        pixels = image.getPrimaryPixels()
        size_z = image.getSizeZ()
        size_c = image.getSizeC()
        size_t = image.getSizeT()
        size_y = image.getSizeY()
        size_x = image.getSizeX()
        z, t, c = 0, 0, 0  # first plane of the image

        zct_list = []

        # For timepoint in T (usually T = 1)
        for t in range(size_t):
            # For the number of Z-stacks
            for z in range(size_z):
                # For each channel in this specific Z-stack
                for c in range(size_c):
                    zct_list.append((z, c, t))

        values = []

        # Load all the planes as YX numpy array
        planes = pixels.getPlanes(zct_list)
        j = 0
        k = 0
        tmp_c = []
        tmp_z = []
        s = "z:%s t:%s c:%s y:%s x:%s" % (size_z, size_t, size_c, size_y, size_x)
        print(s)
        # axis tzyxc
        print("Downloading image %s" % image.getName())
        for i, p in enumerate(planes):
            if k < size_z:
                if j < size_c:
                    tmp_c.append(p)
                    j = j + 1
                if j == size_c:
                    # use dstack to have c at the end
                    tmp_z.append(np.dstack(tmp_c))
                    tmp_c = []
                    j = 0
                    k = k + 1
            if k == size_z:  # done with the stack
                values.append(np.stack(tmp_z))
                tmp_z = []
                k = 0

        return np.stack(values)


def main(args):
    # Argument parser
    parser = argparse.ArgumentParser()

    # ----- Data Saving ----- #

    parser.add_argument(
        "--save_path",
        type=str,
        default="data/IDR/",
        help="Path to save the images",
    )

    parser.add_argument(
        "--metadata_filepath",
        type=str,
        default="metadata.json",
        help="Path to the json file containing the metadata",
    )

    # ----- Data Downloading ----- #

    parser.add_argument(
        "--num_images_requested",
        type=int,
        default=1285,
        help="Number of projects to download",
    )

    # ----- Random ----- #
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Whether to normalize the images",
    )

    parser.add_argument(
        "--resize_size",
        type=int,
        default=0,
        help="If requested, RESIZE each channel of the images to this size",
    )

    parser.add_argument(
        "--crop_size",
        type=int,
        default=0,
        help="If requested, CROP each channel of the images to this size from the center",
    )

    parser.add_argument(
        "--stack_channels",
        action="store_true",
        help="Whether to stack channels or save channel per channel",
    )

    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "npz"],
        help="Format to download the images",
    )

    parser.add_argument(
        "--use_multi_cpus",
        action="store_true",
        help="Whether to use multiple CPUs for downloading",
    )

    parser.add_argument(
        "--max_num_cpus",
        type=int,
        default=1,
        help="Maximum number of CPUs to use for downloading",
    )

    args = parser.parse_args()

    # Start time
    start_time = time.time()

    data_downloader = DataDownloader(
        metadata_filepath=args.metadata_filepath,
        file_format=args.format,
        save_path=args.save_path,
        use_multi_cpus=args.use_multi_cpus,
        max_num_cpus=args.max_num_cpus,
    )

    data_downloader.download(
        num_images_requested=args.num_images_requested,
        normalize=args.normalize,
        resize_size=args.resize_size,
        crop_size=args.crop_size,
        stack_channels=args.stack_channels,
    )

    print(f"Images saved (.{args.format} format) at {args.save_path}.")

    # End time
    end_time = time.time()

    # Print time
    print(f"Time elapsed: {(end_time - start_time)/60} minutes")


if __name__ == "__main__":
    main(sys.argv[1:])
