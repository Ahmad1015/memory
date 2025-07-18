# captioner.py
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# Load model once
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", 
                                                      torch_dtype=torch.float32,
                                                      use_safetensors=True)

def generate_caption(pil_image):
    """
    Generate a caption for a PIL image using BLIP.
    """
    inputs = processor(images=pil_image, return_tensors="pt")

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=20)

    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption
