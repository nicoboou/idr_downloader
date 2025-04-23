import asyncio
import httpx
import streamlit as st

# Base URL for the IDR
IDR_BASE_URL = 'https://idr.openmicroscopy.org'

# Function to display an image given its image_id
async def display_image(image_id):
    IMAGE_DETAILS_URL = "{base}/webclient/imgData/{image_id}/"
    RENDER_IMAGE = "{base}/webgateway/render_image/{image_id}/0/0/"
    qs = {'base': IDR_BASE_URL, 'image_id': image_id}
    url = IMAGE_DETAILS_URL.format(**qs)
    async with httpx.AsyncClient() as client:
        r = await client.get(url)
        if r.status_code == 200:
            img_url = RENDER_IMAGE.format(**qs)
            st.markdown("<img src='%s' width='300' height='300' />" % img_url, unsafe_allow_html=True)
            st.write(f"**Image ID**: {image_id}")
            #st.image(img_url)

async def display_images(random_image_ids):
    await asyncio.gather(*(display_image(image_id) for image_id in random_image_ids))

# Set the random seed for reproducibility
#random.seed(42)

# Function to retrieve image_ids from "screens"
def get_screen_image_ids(metadata):
    screen_image_ids = []
    for screen in metadata["screens"].values():
        for well in screen["plates"].values():
            for field in well["fields"].values():
                screen_image_ids.append(field["image_id"])
    return screen_image_ids

# Function to retrieve image_ids from "projects"
def get_project_image_ids(metadata):
    project_image_ids = []
    for project in metadata["projects"].values():
        for dataset in project["datasets"].values():
            for image in dataset["images"].values():
                project_image_ids.append(image["image_id"])
    return project_image_ids