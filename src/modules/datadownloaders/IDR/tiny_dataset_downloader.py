# Standard imports
import os
import sys
import argparse
import random
from typing import Union
import time

# Third party libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
from tqdm import tqdm

import omero
from idr import connections

dir_path = os.getcwd()
sys.path.insert(0, dir_path)

# Local imports
from src.modules.datadownloaders.IDR.data_downloader import DataDownloader
    
class TinyDatasetDownloader(DataDownloader):
    """
    Class to download tiny dataset from IDR.

    Attributes
    ----------
    metadata_filepath (str): Path to the metadata.json file
    """

    def __init__(self, metadata_filepath:str):
        self.metadata_filepath = metadata_filepath
        
    def download_tiny_dataset(self, num_images_to_keep:int, normalize:bool=True) -> Union[list, list]:
        """
        Download images from IDR

        Parameters
        ----------
        num_images_to_keep (int): Number of images to keep per experiment
        normalize (bool): Normalize images or not

        Returns
        -------
        tiny_dataset_dict (dict): Dictionary of images numpy arrays
        histogram_intensities (dict): Dictionary of histogram intensities per channel
        representative_image_ids (dict): Dictionary of representative image ids per experiment
        empty_images (list): List of empty images
        """

        # Instanciate tiny dataset list
        tiny_dataset_dict = {}

        # Get image ids from metadata (unique phenotypes OR random selection)
        #image_ids = preselected_image_ids[metadata_file]
        image_ids = self.get_image_ids(metadata_path=self.metadata_filepath, num_projects=1, num_screens=1, seed=42)

        # Download images and retrieve their histograms per channel IF NOT EMPTY (Otsu Threshold)
        images_numpy, histogram_intensities, filtered_image_ids, empty_images = self.retrieve_intensity_histograms_from_image_ids(image_ids=image_ids, normalize=normalize)

        # Keep n images_id with highest Bhattacharyya distance
        representative_image_ids = self.keep_images_with_different_histograms(histogram_intensities=histogram_intensities, num_images_to_keep=num_images_to_keep)

        # Retrieve images numpy arrays from images_numpy dict
        for experiment_id, _ in tqdm(images_numpy.items()):
            tiny_dataset_dict[experiment_id] = {}
            for image_id in tqdm(representative_image_ids[experiment_id]):
                if image_id in images_numpy[experiment_id].keys():
                    # Append image numpy array to the list
                    tiny_dataset_dict[experiment_id][image_id] = images_numpy[experiment_id][image_id]
        
        return tiny_dataset_dict, histogram_intensities, representative_image_ids, empty_images
    
    def get_image_ids(self, metadata_path: str, num_projects: int, num_screens: int, seed: int) -> list:
        """
        Retrieve representative image ids from IDR according to the following criteria:
        
        - Check phenotype per image => must have maximum 3 images per phenotype
        - OR IF NO PHENOTYPE IN EXPERIMENT => randomly select maximum 20 images per experiment

        Parameters
        ----------
        metadata_path (str): Path to the metadata.json file
        num_projects (int): Number of projects to select
        num_screens (int): Number of screens to select
        seed (int): Seed for random selection

        Returns
        -------
        image_ids (list): List of image ids
        """

        # Load metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        # Set seed
        random.seed(seed)

        image_ids = {}
        already_queried_phenotypes = {}

        projects = metadata.get('projects', {})
        screens = metadata.get('screens', {})
        project_ids = list(projects.keys())
        screen_ids = list(screens.keys())

        # Randomly select projects
        if len(project_ids) > 0:
            random_project_ids = random.sample(project_ids, min(num_projects, len(project_ids)))

            # For each project in ramdomly selected projects
            for project_id in random_project_ids:

                #  Instanciate image_ids list for this project
                experiment_id = 'project_' + str(projects[project_id].get('experiment_id', {}))
                image_ids[experiment_id] = []

                # Counter for images appended to the list from this project
                images_appended = 0

                datasets = projects[project_id].get('datasets', {})
                nb_datasets = projects[project_id].get('total_datasets', {})
                for dataset in datasets.values():
                    images = dataset.get('images', {})
                    for image in images.values():
                        # Check if phenotype key exists
                        if 'Phenotype Term Accession' in image['image_metadata'].keys():
                            phenotype = image["image_metadata"]['Phenotype Term Accession']
                            if phenotype not in already_queried_phenotypes.keys():
                                already_queried_phenotypes[phenotype] = 1
                            else:
                                already_queried_phenotypes[phenotype] += 1
                            if already_queried_phenotypes[phenotype] < 3:
                                image_ids[experiment_id].append(image['image_id'])
                                images_appended += 1
                
                # If no images were appended, randomly select 20 images from this project
                if images_appended == 0:

                    print(f'No specific phenotype, appending random images from {experiment_id}')

                    # Randomly select max 5 datasets from this project
                    random_dataset_ids = random.sample(list(datasets.keys()), min(5, len(datasets.keys())))
                    
                    # For each dataset in randomly selected datasets
                    for dataset_id in random_dataset_ids:

                        # Randomly select max 4 image_ids from this dataset
                        random_image_ids = random.sample(list(datasets[dataset_id]['images'].keys()), min(4, len(datasets[dataset_id]['images'].keys())))
                        image_ids[experiment_id].extend(datasets[dataset_id]['images'][random_image_id]['image_id'] for random_image_id in random_image_ids)

            print(f'Number of images selected from {experiment_id}: {len(image_ids[experiment_id])}')

        # Randomly select screens
        if len(screen_ids) > 0:
            random_screen_ids = random.sample(screen_ids, min(num_screens, len(screen_ids)))

            # For each screen in ramdomly selected screens
            for screen_id in random_screen_ids:

                #  Instanciate image_ids list for this project
                experiment_id = 'screen_' + str(screens[screen_id].get('experiment_id', {}))
                image_ids[experiment_id] = []

                # Counter for images appended to the list from this screen
                images_appended = 0

                plates = screens[screen_id].get('plates', {})
                nb_plates = screens[screen_id].get('total_plates', {})

                for plate in plates.values():
                    wells = plate.get('wells', {})
                    for well in wells.values():
                        fields = well.get('fields', {})
                        for field in fields.values():
                            # Check if phenotype key exists
                            if 'Phenotype Term Accession' in field['image_metadata'].keys():
                                phenotype = field['image_metadata']['Phenotype Term Accession']
                                if phenotype not in already_queried_phenotypes.keys():
                                    already_queried_phenotypes[phenotype] = 1
                                else:
                                    already_queried_phenotypes[phenotype] += 1
                                if already_queried_phenotypes[phenotype] < 3:
                                    image_ids[experiment_id].append(field['image_id'])
                                    images_appended += 1

                # If no images were appended, randomly select 20 images from this screen
                if images_appended == 0:

                    print(f'No specific phenotypee, appending random images from {experiment_id}')

                    # Randomly select max 7 plates from this screen
                    random_plate_ids = random.sample(list(plates.keys()), min(7, len(plates.keys())))

                    # For each plate in randomly selected plates
                    for plate_id in random_plate_ids:

                        # Randomly select max 3 wells from this plate
                        random_well_ids = random.sample(list(plates[plate_id]['wells'].keys()), min(3, len(plates[plate_id]['wells'].keys())))

                        # For each well in randomly selected wells
                        for well_id in random_well_ids:

                            # Randomly select max 1 image_ids from this well
                            random_image_ids = random.sample(list(plates[plate_id]['wells'][well_id]['fields'].keys()), min(1, len(plates[plate_id]['wells'][well_id]['fields'].keys())))
                            image_ids[experiment_id].extend(plates[plate_id]['wells'][well_id]['fields'][random_image_id]['image_id'] for random_image_id in random_image_ids)

            print(f'Number of images selected from {experiment_id}: {len(image_ids[experiment_id])}')

        return image_ids

    def retrieve_intensity_histograms_from_image_ids(self, image_ids: dict, required_min_otsu_threshold: float = 0.01, required_max_otsu_threshold: float = 0.5, normalize:bool=True) -> Union[dict, dict]:
        """
        Retrieve the intensity histograms from a list of image ids.
        
        For each image id in the list:
            1. Download image and convert it to numpy array
            2. Check if image is empty with Otsu thresholding
            3. If image is not empty, retrieve its histogram intensities for each channel
    
        Parameters
        ----------
        image_ids (list): List of image ids
        required_min_otsu_threshold (float): Required minimum Otsu threshold for an image to be considered empty
        required_max_otsu_threshold (float): Required maximum Otsu threshold for an image to be considered empty

        Returns
        -------
        images_numpy (dict): Dictionary containing the images numpy arrays
        histogram_intensities (dict): Dictionary containing the histogram intensities of each image
        representative_image_ids (list): List of non empty images selected
        empty_images (list): List for storing the number of images that were empty
        """

        # Dict for storing images numpy arrays
        images_numpy = {}

        # Dict for storing histogram intensities
        histogram_intensities = {}

        # List of non empty images selected
        representative_image_ids = []

        # List for storing the number of images that were empty
        empty_images = []

        # 1. Download images and retrieve their histograms
        for experiment_id, val in tqdm(image_ids.items()):

            # Instanciate dicts
            images_numpy[experiment_id] = {}
            histogram_intensities[experiment_id] = {}

            # For each image id in the list
            for image_id in tqdm(image_ids[experiment_id]):

                # Download image and convert it to numpy array
                image_numpy, channels, size_z, size_c, size_t, size_y, size_x = super().download_image(image_id=image_id, normalize=normalize)

                # Instantiate dict for numpy arrays
                images_numpy[experiment_id][image_id] = image_numpy

                # Check if image is empty
                is_empty, mean_foreground_percentage  = TinyDatasetDownloader.check_if_image_is_empty(image=image_numpy, required_min_threshold=required_min_otsu_threshold, required_max_threshold=required_max_otsu_threshold)
                
                # If image is not empty, calculate the histogram
                if not is_empty:

                    # Instantiate dict for histogram intensities
                    histogram_intensities[experiment_id][image_id] = {}

                    for channel in range(size_c):
                        # Calculate the histogram of the image because it's not too dark
                        hist = TinyDatasetDownloader.calculate_intensity_histogram(image=image_numpy, channel_number=channel, channel_name=channels[channel].getName(), plot=False)
                        histogram_intensities[experiment_id][image_id][channel] = hist

                    representative_image_ids.append(image_id)
                
                # If image is empty, add it to the list of empty images
                else:
                    empty_images.append(image_id)

        return images_numpy, histogram_intensities, representative_image_ids, empty_images
    
    def keep_images_with_different_histograms(self, histogram_intensities: dict, num_images_to_keep: int = 2) -> list:
        """
        Keep images with different histograms

        Parameters
        ----------
        histogram_intensities (dict): Dictionary containing the histogram intensities of each image
        threshold (float): Threshold for the histogram distance. If the distance is above the threshold, the images are considered to be different, and vice versa.
        n (int): Number of image ids to keep with the highest distances.

        Returns
        -------
        representative_ids (list): List of image ids with different histograms
        """
        # Compare histogram intensities
        representative_ids = {}
        distances = {}

        # For each experiment
        for experiment_id, val in histogram_intensities.items():
            
            # Instantiate dict
            distances[experiment_id] = {}
            
            # For each image in the experiment
            for image_id, hist_intensity in tqdm(histogram_intensities[experiment_id].items()):

                # Get histogram intensities
                histograms_to_check_1 = hist_intensity

                # Compare with all other images
                for other_image_id in histogram_intensities[experiment_id].keys():
                    # Skip if same image
                    if image_id == other_image_id:
                        continue

                    # Get histogram intensities
                    histograms_to_check_2 = histogram_intensities[experiment_id][other_image_id]

                    # Compare histograms
                    mean_distance = self.compute_bhattacharyya_distance(histograms_to_check_1, histograms_to_check_2)

                    print(f"Distance between images {image_id} and {other_image_id}: {mean_distance}")
                    
                    # Store the distance for both images in the dictionary
                    if image_id not in distances[experiment_id]:
                        distances[experiment_id][image_id] = []
                    distances[experiment_id][image_id].append(mean_distance)

                    if other_image_id not in distances:
                        distances[experiment_id][other_image_id] = []
                    distances[experiment_id][other_image_id].append(mean_distance)

            # If the loop was not broken, calculate the average distance for the image by iterating over the distances dictionary
            for image_id, distance in distances[experiment_id].items():
                distances[experiment_id][image_id] = np.mean(distance)

        # Sort the distances dictionary based on average distances in descending order
        sorted_distances = {
            key: dict(sorted(inner_dict.items(), key=lambda item: item[1], reverse=True))
            for key, inner_dict in distances.items()
        }

        print(f"Averaged sorted distances: {sorted_distances}")

        # Take the first n image ids from the sorted dictionary
        for experiment_id, inner_dict in distances.items():
            sorted_images = sorted(inner_dict.items(), key=lambda item: item[1], reverse=True)
            top_n_images = [image_id for image_id, _ in sorted_images[:min(num_images_to_keep, len(sorted_images))]]
            representative_ids[experiment_id] = top_n_images

        return representative_ids


    def download_image(self, image_id:int, normalize:bool = True) -> np.ndarray:
        return super().download_image(image_id, normalize=normalize)

    def download_image_with_size_limit(self, image_id: int, target_width: int = 224, target_height: int = 224):
        return super().download_image_with_size_limit(image_id, target_width, target_height)
    
    @staticmethod
    def compute_otsu_threshold(channel: np.ndarray):
        """
        Compute the Otsu threshold for a given channel of an image

        Parameters
        ----------
        image (np.ndarray): Image to compute the threshold for
        channel_number (int): Channel number
        channel_name (str): Channel name
        plot (bool): Whether to plot the histogram and the threshold or not

        Returns
        -------
        threshold (float): Otsu threshold
        """

        # Compute the Otsu threshold
        threshold_value, thresholded_channel = cv2.threshold(channel,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        return threshold_value, thresholded_channel

    @staticmethod
    def check_if_image_is_empty(image: np.ndarray, required_min_threshold: float = 0.01, required_max_threshold: float = 0.45):
       
        """
        Check if an image is empty using Otsu's thresholding.
        
        Args:
            image (numpy.ndarray): Input image.
        
        Returns:
            bool: True if the image is empty, False otherwise.
        """
        
        # List of foreground percentages
        foreground_percentages = []

        if len(image.shape) == 3:
            for channel in range(image.shape[2]):
                # Extract the channel
                channel = image[:,:,channel]
                # Apply Otsu's thresholding
                _, thresholded_channel = TinyDatasetDownloader.compute_otsu_threshold(channel=channel)
                
                # Calculate the percentage of foreground pixels
                foreground_percentage = np.count_nonzero(thresholded_channel) / (channel.shape[0] * channel.shape[1])
                foreground_percentages.append(foreground_percentage)
        
        elif len(image.shape) == 2:
            # Apply Otsu's thresholding
            _, thresholded_channel = TinyDatasetDownloader.compute_otsu_threshold(channel=image)
            
            # Calculate the percentage of foreground pixels
            foreground_percentage = np.count_nonzero(thresholded_channel) / (image.shape[0] * image.shape[1])
            foreground_percentages.append(foreground_percentage)

        # Check if the image is empty based on the foreground percentage
        if np.mean(foreground_percentages) > required_min_threshold and np.mean(foreground_percentages) < required_max_threshold:
            return False, np.mean(foreground_percentages)
        
        else:
            return True, np.mean(foreground_percentages)

    @staticmethod
    def calculate_intensity_histogram(image, channel_number=0, channel_name:str='DNA', plot:bool=False):
        """
        Calculate the intensity histogram of a specified channel in an image using the original number of bins.
        
        Args:
            image (numpy.ndarray): Input image.
            channel (int): Channel index to calculate the histogram for. Default is 0.
        
        Returns:
            numpy.ndarray: Intensity histogram.
        """
        # Extract the specified channel
        channel_image = image[:, :, channel_number]

        # Flatten the image into a 1D array
        pixel_values = channel_image.flatten()

        # Get unique pixel values and their frequencies
        unique_values, pixel_counts = np.unique(pixel_values, return_counts=True)

        # Determine the number of unique pixel values
        num_unique_values = len(unique_values)

        # Set the bin edges based on the unique pixel values
        bin_edges = np.arange(num_unique_values + 1)

        if plot:
            # Plot the pixel values against their frequencies
            plt.hist(pixel_values, bins=bin_edges)
            plt.xlabel('Pixel Value')
            plt.ylabel('Frequency')
            plt.title(channel_name + " intensity histogram")
            plt.show()
        
        return channel_image.flatten()

    @staticmethod
    def compute_jaccard_distance(histograms_1, histograms_2):
        """
        Compute the histogram intersection distance between histograms of each channel.
        WARNING: The histogram intersection algorithm was proposed by Swain and Ballard in their article “Color Indexing” This algorithm is ONLY particular reliable when the color is a strong predictor of the object identity !!!!
        The value distribution is a way to represent the colour appearance and in the HVS model it represents the saturation of a colour.
        
        BENEFITS: The histogram intersection does not require the accurate separation of the object from its background and it is robust to occluding objects in the foreground.
        Histograms are invariant to translation and they change slowly under different view angles, scales and in presence of occlusions.

        Args:
            histograms_1 (list): List of histograms for each channel of the first image.
            histograms_2 (list): List of histograms for each channel of the second image.
        
        Returns:
            float: Histogram intersection distance.
        """
        # Calculate the histogram intersection distance for each channel
        distances = []

        for index, channel in enumerate(range(len(histograms_1))):
            # Calculate the histogram intersection distance
            intersection = np.minimum(histograms_1[channel], histograms_2[channel]).sum()
            # Normalize the distance by the total number of pixels
            distance = 1.0 - (intersection / histograms_1[channel].sum())
            distances.append(distance)
        
        # Return the mean distance
        return np.mean(distances)

    @staticmethod
    def compute_bhattacharyya_distance(histograms_1, histograms_2):
        """
        Compute the Bhattacharyya distance between two histograms.
        The Bhattacharyya distance is a non-negative value, with smaller distances indicating higher similarity and larger distances indicating greater dissimilarity between the distributions.
        
        Args:
            hist1 (numpy.ndarray): First histogram.
            hist2 (numpy.ndarray): Second histogram.
        
        Returns:
            float: Mean  Bhattacharyya distance.
        """
        # Calculate the histogram intersection distance for each channel
        distances = []

        # If there IS the same number of channels, the images are comparable
        if len(histograms_1) == len(histograms_2):

            for i, channel in enumerate(range(len(histograms_1))):

                # Assert that the histograms have the same number of bins
                if histograms_1[channel].shape[0] == histograms_2[channel].shape[0]: # The histograms must have the same number of bins, theycan be compared

                    # Normalize the histograms
                    hist1 = histograms_1[channel].astype(float) / np.sum(histograms_1[channel])
                    hist2 = histograms_2[channel].astype(float) / np.sum(histograms_2[channel])

                    # Compute the Bhattacharyya coefficient
                    b_coeff = np.sum(np.sqrt(hist1 * hist2))
                    
                    # Compute the Bhattacharyya distance
                    b_distance = -np.log(b_coeff)

                    distances.append(b_distance)
                else:
                    print("The histograms do not have the same number of bins => they are significantly different.")
                    distances.append(1.0)

        # If there is NOT the same number of channels, the images are NOT comparable => they are different
        else:
            print("The histograms do not have the same number of channels => they are significantly different.")
            distances.append(1.0)

        # Compute the mean distance
        mean_distance = np.mean(distances)

        return mean_distance
        
def main(args):

    # Argument parser
    parser = argparse.ArgumentParser()

    # ----- Data Saving ----- #

    parser.add_argument(
        "--save_path",
        type=str,
        default='data/IDR/cell/tiny_dataset',
        help="Directory to save the images",
    )

    parser.add_argument(
        "--metadata_filepath",
        type=str,
        default='data/cell/meta',
        help="Path to the json file containing the metadata",
    )

    parser.add_argument(
        "--num_images_to_keep",
        type=int,
        default=2,
        help="Number of images to keep in the tiny dataset"
    )

    parser.add_argument(
        "--normalize",
        action='store_true',
        help="Whether to normalize the images or not"
    )

    args = parser.parse_args()

    # Download images
    tiny_dataset_downloader = TinyDatasetDownloader(metadata_filepath=args.metadata_filepath)
    tiny_dataset_dict, _, _, _ = tiny_dataset_downloader.download_tiny_dataset(num_images_to_keep=args.num_images_to_keep, normalize=args.normalize)

    for experiment_id, experiment_dict in tiny_dataset_dict.items():
        # If the experiment is not empty, let's save it
        if bool(experiment_dict):
            # Retrieve the list of images
            experiment_list, representative_image_ids = list(experiment_dict.values()), list(experiment_dict.keys())

            # Save tiny dataset to npz file
            filename = args.save_path + 'tiny_dataset_' + str(experiment_id) + '.npz'
            DataDownloader.save_to_npz(images=experiment_list, labels=representative_image_ids, filename=filename)

if __name__ == "__main__":
    main(sys.argv[1:])