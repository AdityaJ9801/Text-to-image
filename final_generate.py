import requests
import io
from PIL import Image
import torch
from IPython import display


API_TOKEN = "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"

API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

device ="cpu" # "cuda" if torch.cuda.is_available() else "cpu"

model = torch.hub.load("bryandlee/animegan2-pytorch:main", "generator", device=device).eval()

face2paint = torch.hub.load("bryandlee/animegan2-pytorch:main", "face2paint", device=device)

def query(payload):
    """
    Sends a POST request to the Hugging Face API with the provided payload.

    Args:
        payload (dict): The payload containing the request inputs and options.

    Returns:
        bytes: The content of the response.
    """
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.content

def text_to_image(prompt, image_name):
    """
    Converts a text prompt into an image using the Hugging Face API.

    Args:
        prompt (str): The text prompt to generate an image from.
        image_name (str): The desired name for the generated image.

    Returns:
        PIL.Image.Image: The generated image as a PIL Image object.
    """
    image_bytes = query({"inputs" : prompt,
                         "options" : {"use_cache" : False,
                                      "wait_for_model" : True}})
    image_raw = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_raw.save(image_name + ".png")
    return image_raw

def filter_run(image, image_name):
    """
    Applies the Face2Paint filter to the input image.

    Args:
        image (PIL.Image.Image): The input image to apply the filter to.
        image_name (str): The desired name for the filtered image.

    Returns:
        PIL.Image.Image: The filtered image as a PIL Image object.
    """
    im_in = image
    im_out = face2paint(model, im_in, side_by_side=False)
    im_out.save(image_name + "_filtered" + ".png")
    return im_out

def run_promp(prompt, image_name):
    """
    Generates and filters images based on the provided prompt.

    Args:
        prompt (str): The text prompt to generate an image from.
        image_name (str): The desired name for the generated and filtered images.

    Returns:
        Tuple[PIL.Image.Image, PIL.Image.Image]: A tuple containing the generated and filtered images as PIL Image objects.
    """

    image_raw = text_to_image(prompt, image_name)
    image_filter = filter_run(image_raw, image_name)

    return image_raw, image_filter

def generate_and_save_image(prompt, file_name, save_image=True):
    """
    Generates and displays an image based on the provided prompt and saves it to the disk.

    Args:
        prompt (str): The text prompt to generate an image from.
        file_name (str): The desired name for the generated and filtered images.
        save_image (bool, optional): Whether to save the images to the disk. Defaults to True.

    Returns:
        None
    """
    image_raw, image_filter = run_promp(prompt, file_name)
    print("\n #### Raw image #### \n")
    display.display(image_raw)
    print("\n #### Filtered image #### \n")
    display.display(image_filter)

    if save_image:
        image_raw.save(file_name + ".png")
        image_filter.save(file_name + "_filtered" + ".png")

# Example usage:
prompt = "all country union"
file_name = "nation"
generate_and_save_image(prompt, file_name, save_image=True)

prompt = "galaxy in outer space"
file_name = "galaxy"
generate_and_save_image(prompt, file_name, save_image=True)
