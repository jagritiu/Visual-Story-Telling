


import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import os
import re

# Load BLIP model and processor for image captioning
blip_model_path = "models/blip-finetuned"  # Adjust the path to your fine-tuned BLIP model
blip_processor = BlipProcessor.from_pretrained(blip_model_path)
blip_model = BlipForConditionalGeneration.from_pretrained(blip_model_path)

# Load T5 model and tokenizer for story generation
model_dir = "./story_model"  # Directory where your fine-tuned T5 model is saved
story_tokenizer = T5Tokenizer.from_pretrained(model_dir)
story_model = T5ForConditionalGeneration.from_pretrained(model_dir)

# Move models to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
blip_model.to(device)
story_model.to(device)

# Function to generate caption for a single image using BLIP model
def generate_caption(image_path):
    """
    Generate a caption for a single image using BLIP model
    """
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    inputs = blip_processor(images=image, return_tensors="pt").to(device)

    # Generate caption
    blip_model.eval()
    with torch.no_grad():
        output = blip_model.generate(**inputs)

    caption = blip_processor.decode(output[0], skip_special_tokens=True)
    return caption

# Function to generate captions for a list of image IDs
def generate_captions_for_images(image_paths):
    """
    Generate captions for a list of images.
    """
    captions = []
    for image_path in image_paths:
        caption = generate_caption(image_path)
        captions.append(caption)
        print(f"Generated Caption for {image_path}: {caption}")  # Debugging statement
    return captions

# Function to generate a story from the combined caption
def generate_story(input_text, model, tokenizer, device):
    """
    Generate a story from the combined input text using the T5 model.
    """
    model.to(device)
    model.eval()

    # Clean up input text
    input_text = re.sub(r'No generated caption found', '', input_text)
    input_text = re.sub(r'\s+', ' ', input_text).strip()

    # Prepare input for model
    input_text = f"generate story: {input_text}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    # Generate output
    output_ids = model.generate(
        input_ids,
        max_length=200,
        num_beams=4,
        no_repeat_ngram_size=2,
        early_stopping=True
    )

    # Decode output
    story = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"Generated Story: {story}")  # Debugging statement
    return story

# Tkinter GUI for selecting images and displaying output
class CaptioningApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Captioning and Story Generation")

        # Add a button to select images and generate story near each other
        self.upload_button = tk.Button(self.root, text="Upload Images", command=self.upload_images, font=("Helvetica", 12))
        self.upload_button.grid(row=0, column=0, columnspan=2, pady=10, padx=20)

        # Add a button to generate the story
        self.story_button = tk.Button(self.root, text="Generate Story", command=self.generate_story_ui, font=("Helvetica", 12))
        self.story_button.grid(row=0, column=2, pady=10, padx=20)

        # Add a frame to hold images and captions
        self.image_frame = tk.Frame(self.root)
        self.image_frame.grid(row=1, column=0, columnspan=3, padx=20, pady=20)

        # Add a text area to display the story
        self.story_heading = tk.Label(self.root, text="Generated Story:", font=("Helvetica", 14))
        self.story_heading.grid(row=2, column=0, columnspan=3, pady=10)

        self.story_text = tk.Text(self.root, height=10, width=100, font=("Helvetica", 12))
        self.story_text.grid(row=3, column=0, columnspan=3, padx=20, pady=20)

        self.image_paths = []

    def upload_images(self):
        """Open file dialog to select images."""
        self.image_paths = filedialog.askopenfilenames(filetypes=[("Image Files", "*.jpg;*.png")])

        if self.image_paths:
            self.display_images_and_captions()

    def display_images_and_captions(self):
        """Display the images in a single row, captions below each image in the next row."""
        # Clear any previous images and captions
        for widget in self.image_frame.winfo_children():
            widget.destroy()

        # Generate captions
        captions = generate_captions_for_images(self.image_paths)

        # Display images in a row and captions in the next row
        for idx, image_path in enumerate(self.image_paths):
            # Load and resize the image to make it bigger
            image = Image.open(image_path)
            image = image.resize((250, 250))  # Increase image size for better visibility
            img = ImageTk.PhotoImage(image)

            # Create image display label
            image_label = tk.Label(self.image_frame, image=img)
            image_label.image = img  # Keep reference to the image object
            image_label.grid(row=0, column=idx, padx=10, pady=10)

            # Create caption label with a larger font size
            caption = captions[idx]
            caption_label = tk.Label(self.image_frame, text=caption, wraplength=200, font=("Helvetica", 12))
            caption_label.grid(row=1, column=idx, padx=10, pady=10)

    def generate_story_ui(self):
        """Generate and display the story."""
        if not self.image_paths:
            messagebox.showwarning("No Images", "Please upload images first!")
            return

        captions = generate_captions_for_images(self.image_paths)
        combined_caption = " ".join(captions)
        story = generate_story(combined_caption, story_model, story_tokenizer, device)

        # Display the story in the text area
        self.story_text.delete(1.0, tk.END)  # Clear the text area before showing new story
        self.story_text.insert(tk.END, story)

# Main function to start the app
def main():
    root = tk.Tk()
    app = CaptioningApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()

