import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import os

# Load BLIP model and processor
model_path = "models/blip-finetuned"  # Adjust the path to your fine-tuned model
processor = BlipProcessor.from_pretrained(model_path)
model = BlipForConditionalGeneration.from_pretrained(model_path)

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def generate_caption(image_path):
    """
    Generate a caption for a single image using BLIP model
    """
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    # Generate caption
    model.eval()
    with torch.no_grad():
        output = model.generate(**inputs)
    
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

def generate_captions_for_images(image_ids, image_folder="images"):
    """
    Generate captions for a list of 5 image IDs.
    """
    captions = []
    for image_id in image_ids:
        image_path = os.path.join(image_folder, f"{image_id}.jpg")
        caption = generate_caption(image_path)
        captions.append(caption)
    return captions

# Example usage: Generate captions for 5 images
image_ids = [
    "1",  # Replace with actual image IDs
    "2",
    "3",
    "4",
    "5"
]

captions = generate_captions_for_images(image_ids)
for i, caption in enumerate(captions, 1):
    print(f"Generated Caption for Image {image_ids[i-1]}: {caption}")
