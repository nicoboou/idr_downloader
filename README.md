# IDR Downloader 

Smol library to download images from the [IDR](https://idr.openmicroscopy.org/) (Image Data Resource) using the [IDR API](https://idr.openmicroscopy.org/api-docs/).
<p align="center">
  <img src="https://github.com/nicoboou/idr_downloader/blob/main/.github/idr_downloader.png" alt="IDR Downloader Overview" width="500"/>
</p>

## Features

- Metadata download from IDR experiments, projects and screens
- Image download with various preprocessing options (normalization, resizing, etc.)
- Tiny dataset creation with diverse image selection
- Interactive data explorer with Streamlit UI

## Installation

### Prerequisites

- Python 3.8+
- Connection to the internet to access IDR API
- Conda or Mamba package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/nicoboou/idr_downloader.git
cd idr_downloader

# Install dependencies
conda env create -f environment.yaml
conda activate idr-downloader 
```

## Usage

### Downloading Metadata

```python
from src.modules.datadownloaders.IDR.metadata_downloader import MetaDataDownloader

# Initialize the metadata downloader
metadata_downloader = MetaDataDownloader()

# Launch a session to connect to IDR
metadata_downloader.launch_session()

# Get all experiment IDs
metadata_downloader.get_experiments_ids()

# Download metadata for a specific experiment
metadata_downloader.dump_json(experiment_id=5514)
```

### Downloading Images

```python
from src.modules.datadownloaders.IDR.data_downloader import DataDownloader

# Initialize the data downloader with metadata file
downloader = DataDownloader(metadata_filepath="path/to/metadata.json")

# Download images
images = downloader.download(
    num_images_requested=10, 
    normalize=True,
    resize_size=512,
    crop_size=0,
    stack_channels=False
)
```

### Creating a Tiny Dataset

```python
from src.modules.datadownloaders.IDR.tiny_dataset_downloader import TinyDatasetDownloader

# Initialize the tiny dataset downloader
tiny_downloader = TinyDatasetDownloader(metadata_filepath="path/to/metadata.json")

# Download a small dataset with diverse images
images = tiny_downloader.download_tiny_dataset(
    num_images_to_keep=10,
    normalize=True
)
```

### Using the Interactive Explorer

The project includes an interactive data explorer built with Streamlit:

```bash
# Navigate to the explorer directory
cd idr_explorer

# Run the Streamlit app
streamlit run idr_explorer.py
```

The explorer provides:
- Overview of available experiments
- Detailed view of experiment metadata
- Image sampling and visualization
- Dataset statistics and distribution analysis

## Project Structure

```
idr_downloader/
├── src/
│   └── modules/
│       └── datadownloaders/
│           └── IDR/
│               ├── data_downloader.py        # Main downloader class
│               ├── metadata_downloader.py    # Metadata retrieval
│               └── tiny_dataset_downloader.py # Optimized small dataset downloader
├── idr_explorer/                             # Interactive UI
│   ├── idr_explorer.py                       # Main Streamlit app
│   ├── screens/                              # UI screens
│   └── funcs/                                # Helper functions
└── figs/                                     # Images and diagrams
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.