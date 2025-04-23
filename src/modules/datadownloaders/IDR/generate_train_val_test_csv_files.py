import os
import sys
import argparse
import random
import csv
from tqdm import tqdm

def generate_train_val_test_csv_files(full_csv_file_path:str, val_size=0.2, test_size=0.2, train_file='train.csv', val_file='val.csv', test_file='test.csv'):
    """
    Takes a full csv file and splits it into train, val and test csv files.
    """
    # Read the full csv file
    with open(full_csv_file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        file_data = [row for row in reader]

    # Calculate the sizes for train, val, and test sets
    total_samples = len(file_data)
    val_samples = int(val_size * total_samples)
    test_samples = int(test_size * total_samples)
    train_samples = total_samples - val_samples - test_samples

    # Shuffle the file_data randomly
    random.shuffle(file_data)

    # Split the file_data into train, val, and test sets
    train_data = file_data[:train_samples]
    val_data = file_data[train_samples:train_samples + val_samples]
    test_data = file_data[train_samples + val_samples:]

    print(f'Number of images in  train set: {len(train_data)}')
    print(f'Number of images in  val set: {len(val_data)}')
    print(f'Number of images in test set : {len(test_data)}')

    print('Saving train.csv file...')
    # Write train data to train_file
    with open(train_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(train_data)
    print(f'train.csv saved!')

    print('Saving val.csv file...')
    # Write val data to val_file
    with open(val_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(val_data)
    print(f'val.csv saved!')

    print('Saving test.csv file...')
    # Write test data to test_file
    with open(test_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(test_data)
    print(f'test.csv saved!')


def generate_file_paths_csv(root_dir, output_dir, csv_file):
    image_files = sorted(os.listdir(root_dir))

    csv_file = os.path.join(output_dir, csv_file)
    
    written_images = set()  # Track the written images to avoid duplicates

    for image_file in tqdm(image_files):
        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            file_parts = image_file.split('_')

            if len(file_parts) >= 4 and file_parts[-1].lower().endswith('.png'):
                experiment_id = file_parts[1]
                image_id = file_parts[2]
                channel_id = file_parts[3].split('.')[0]
                full_image_name = file_parts[0] + "_" + file_parts[1] + "_" + file_parts[2]

                file_paths = [os.path.join(root_dir, image_file) for image_file in image_files if
                            f'_{experiment_id}_{image_id}_' in image_file and channel_id in image_file]

                if len(file_paths) > 0 and full_image_name not in written_images:
                    writer.writerow([full_image_name] + [file_paths]) # Write image name and channel paths
                    written_images.add(full_image_name)  # Add the image to the set of written images
                

def main(args):

    # Argument parser
    parser = argparse.ArgumentParser()

    # ----- Data Retrieving ----- #

    parser.add_argument(
        "--directory",
        type=str,
        default='data/IDR/cell/images',
        help="Path to the images",
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default='data/IDR/cell/',
        help="Path to save the .csv files",
    )

    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Size of the training set",
    )

    parser.add_argument(
        "--val_size",
        type=float,
        default=0.2,
        help="Size of the validation set",
    )

    args = parser.parse_args()

    #Assuming you already have the list of image IDs, 'ids'
    train_file_path = args.save_path + 'train.csv'
    test_file_path = args.save_path + 'test.csv'
    val_file_path = args.save_path + 'val.csv'

    # Extract unique image ids
    print(f'Extracting image ids from {args.directory}')
    generate_file_paths_csv(root_dir=args.directory, output_dir=args.save_path, csv_file='full.csv')
    
    # Split and save
    print(f'Splitting and saving the full.csv file into train, val, and test csv files')
    csv_file_path = args.save_path + 'full.csv'
    generate_train_val_test_csv_files(full_csv_file_path=csv_file_path,val_size=args.val_size, test_size=args.test_size, train_file=train_file_path, val_file=val_file_path, test_file=test_file_path)
    

if __name__ == "__main__":
    main(sys.argv[1:])