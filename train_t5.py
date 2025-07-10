import os
import torch
import re
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # Import tqdm for progress bars

# Set random seed for reproducibility
torch.manual_seed(42)

# Dataset class for image descriptions to stories
class ImageStoryDataset(Dataset):
    def __init__(self, inputs, outputs, tokenizer, max_len=512):
        self.inputs = inputs
        self.outputs = outputs
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        input_text = f"generate story: {self.inputs[idx]}"
        output_text = self.outputs[idx]
        
        # Tokenize inputs
        input_encoding = self.tokenizer.encode_plus(
            input_text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize outputs
        output_encoding = self.tokenizer.encode_plus(
            output_text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # For loss calculation, replace padding tokens with -100
        labels = output_encoding['input_ids'].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'labels': labels.squeeze()
        }

# Parse training data from file
def parse_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Extract input-output pairs using regex
    pattern = r'input:(.*?)output:(.*?)(?=input:|$)'
    matches = re.findall(pattern, content, re.DOTALL)
    
    inputs = []
    outputs = []
    
    for match in matches:
        input_text = match[0].strip()
        output_text = match[1].strip()
        
        # Clean up input text (remove "No generated caption found")
        input_text = re.sub(r'No generated caption found', '', input_text)
        input_text = re.sub(r'\s+', ' ', input_text).strip()
        
        # Clean up output text
        output_text = re.sub(r'\s+', ' ', output_text).strip()
        
        if input_text and output_text:
            inputs.append(input_text)
            outputs.append(output_text)
    
    return inputs, outputs

# Train the model
def train_model(train_file, output_dir="./story_model", epochs=10):
    # Parse the data
    inputs, outputs = parse_data(train_file)
    print(f"Loaded {len(inputs)} training examples")
    
    # Split into train/validation sets
    train_inputs, val_inputs, train_outputs, val_outputs = train_test_split(
        inputs, outputs, test_size=0.2, random_state=42
    )
    
    # Initialize T5 model and tokenizer
    model_name = "t5-base"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    # Create datasets
    train_dataset = ImageStoryDataset(train_inputs, train_outputs, tokenizer)
    val_dataset = ImageStoryDataset(val_inputs, val_outputs, tokenizer)
    
    # Create dataloaders
    batch_size = 2  # Small batch size due to limited data
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Set up training device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        
        # Use tqdm to display a progress bar for each batch in training
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}")):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Clear gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_train_loss += loss.item()
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            # Use tqdm to display a progress bar for validation
            for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"Validating Epoch {epoch+1}/{epochs}")):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"  Saving model to {output_dir}")
            
            # Create directory if it doesn't exist
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Save model and tokenizer
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
    
    print("Training complete!")
    return model, tokenizer

# Generate stories using the trained model
def generate_story(model, tokenizer, input_text, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    model.eval()
    
    # Clean up input text
    input_text = re.sub(r'No generated caption found', '', input_text)
    input_text = re.sub(r'\s+', ' ', input_text).strip()
    
    # Prepare input
    input_text = f"generate story: {input_text}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    
    # Generate output
    output_ids = model.generate(
        input_ids,
        max_length=100,
        num_beams=4,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    
    # Decode output
    story = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return story

# Main function to run the training and test the model
def main():
    # File paths
    train_file = "train_data.txt"
    model_dir = "./story_model"
    
    # Check if model already exists
    if os.path.exists(model_dir) and os.path.isfile(os.path.join(model_dir, "pytorch_model.bin")):
        print(f"Loading existing model from {model_dir}")
        tokenizer = T5Tokenizer.from_pretrained(model_dir)
        model = T5ForConditionalGeneration.from_pretrained(model_dir)
    else:
        print(f"Training new model...")
        model, tokenizer = train_model(train_file, model_dir)
    
    # Test the model with an example
    test_input = """The tree has very long and dated branches. 
A plaque on a stand surround by died leaves off a tree. 
A huge tree sits outside with several large roots stemming from the trunk. 
A person is taking a picture of a large tree and you can see their shadow 
Various parts of the tree are growing out from the ground."""
    
    story = generate_story(model, tokenizer, test_input)
    
    print("\nGenerated Story:")
    print(story)
    
    # Allow user to generate more stories
    print("\nEnter 'q' to quit or input new image descriptions to generate a story:")
    while True:
        user_input = input("Enter image descriptions (or 'q' to quit): ")
        if user_input.lower() == 'q':
            break
        
        story = generate_story(model, tokenizer, user_input)
        print("\nGenerated Story:")
        print(story)

if __name__ == "__main__":
    main()
