import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import json
import re
import requests
from io import BytesIO

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
def generate_caption(image_url):
    """
    Generate a caption for a single image using BLIP model
    """
    # Download the image from URL
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    inputs = blip_processor(images=image, return_tensors="pt").to(device)

    # Generate caption
    blip_model.eval()
    with torch.no_grad():
        output = blip_model.generate(**inputs)

    caption = blip_processor.decode(output[0], skip_special_tokens=True)
    return caption

# Function to generate story from the combined caption
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
    return story

# Tkinter GUI for selecting images and displaying output
class CaptioningApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Captioning and Story Generation")

        # Add a label and entry for story ID
        self.story_id_label = tk.Label(self.root, text="Enter Story ID:", font=("Helvetica", 14, "bold"))
        self.story_id_label.grid(row=0, column=0, padx=10, pady=10)

        self.story_id_entry = tk.Entry(self.root, font=("Helvetica", 14))
        self.story_id_entry.grid(row=0, column=1, padx=10, pady=10)

        # Add a button to fetch the story and display images
        self.load_button = tk.Button(self.root, text="Load Story", command=self.load_story, font=("Helvetica", 12, "bold"))
        self.load_button.grid(row=0, column=2, padx=10, pady=10)

        # Add a frame to hold images and captions
        self.image_frame = tk.Frame(self.root)
        self.image_frame.grid(row=1, column=0, columnspan=3, padx=20, pady=20)

        # Add a text area to display the story
        self.story_heading = tk.Label(self.root, text="Generated Story:", font=("Helvetica", 16, "bold"))
        self.story_heading.grid(row=2, column=0, columnspan=3, pady=10)

        self.story_text = tk.Text(self.root, height=10, width=100, font=("Helvetica", 14))
        self.story_text.grid(row=3, column=0, columnspan=3, padx=20, pady=20)

        # To hold the story data
        self.story_data = None

    def load_story(self):
        """Load the story based on story_id and display images and captions."""
        story_id = self.story_id_entry.get()

        if not story_id:
            messagebox.showwarning("Empty Story ID", "Please enter a valid story ID!")
            return

        # Load caption.json file
        try:
            with open("caption.json", "r") as f:
                data = json.load(f)
        except Exception as e:
            messagebox.showerror("Error", f"Error loading caption file: {e}")
            return

        # Search for the story_id
        self.story_data = None
        for story in data["stories"]:
            if story["story_id"] == story_id:
                self.story_data = story
                break

        if not self.story_data:
            messagebox.showwarning("Story Not Found", f"No story found with ID {story_id}")
            return

        # Display images and captions
        self.display_images_and_captions()

    def display_images_and_captions(self):
        """Display the images in a single row, captions below each image in the next row."""
        # Clear any previous images and captions
        for widget in self.image_frame.winfo_children():
            widget.destroy()

        # Get the images and captions
        images = self.story_data["images"]
        generated_captions = []
        ground_truth_captions = []

        # Generate captions and display images with their ground truth captions
        for idx, image in enumerate(images):
            # Load the image
            image_url = image["url"]
            image_caption = image["caption"]

            # Generate a caption using BLIP
            generated_caption = generate_caption(image_url)
            generated_captions.append(generated_caption)
            ground_truth_captions.append(image_caption)

            # Load and resize the image to make it bigger
            response = requests.get(image_url)
            img = Image.open(BytesIO(response.content))
            img = img.resize((250, 250))  # Resize image for better visibility
            img_display = ImageTk.PhotoImage(img)

            # Create image display label
            image_label = tk.Label(self.image_frame, image=img_display)
            image_label.image = img_display  # Keep reference to the image object
            image_label.grid(row=0, column=idx, padx=10, pady=10)

            # Ground truth caption label with bold heading, content remains normal
            gt_caption_label = tk.Label(self.image_frame, text=f"Ground Truth:", font=("Helvetica", 12, "bold"))
            gt_caption_label.grid(row=1, column=idx, padx=10, pady=5)

            # Ground truth content label with normal font
            gt_content_label = tk.Label(self.image_frame, text=f"{image_caption}", wraplength=200, font=("Helvetica", 12))
            gt_content_label.grid(row=2, column=idx, padx=10, pady=5)

            # Generated caption label with bold heading, content remains normal
            gen_caption_label = tk.Label(self.image_frame, text=f"Generated Caption:", font=("Helvetica", 12, "bold"))
            gen_caption_label.grid(row=3, column=idx, padx=10, pady=5)

            # Generated content label with normal font
            gen_content_label = tk.Label(self.image_frame, text=f"{generated_caption}", wraplength=200, font=("Helvetica", 12))
            gen_content_label.grid(row=4, column=idx, padx=10, pady=5)

        # Combine generated captions for story generation
        combined_caption = " ".join(generated_captions)
        story = generate_story(combined_caption, story_model, story_tokenizer, device)

        # Display the generated story in the text area
        self.story_text.delete(1.0, tk.END)  # Clear the text area before showing new story
        self.story_text.insert(tk.END, story)

# Main function to start the app
def main():
    root = tk.Tk()
    app = CaptioningApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
