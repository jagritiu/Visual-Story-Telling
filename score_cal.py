import json
import re
import os
from nltk.tokenize import word_tokenize
from rouge import Rouge
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import nltk

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Load T5 model and tokenizer for story generation
model_dir = "./story_model"  # Directory where your fine-tuned T5 model is saved
story_tokenizer = T5Tokenizer.from_pretrained(model_dir)
story_model = T5ForConditionalGeneration.from_pretrained(model_dir)

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
story_model.to(device)

def generate_story(input_text, model, tokenizer, device):
    """
    Generate a story from the input text using the T5 model.
    """
    model.eval()

    # Clean up input text
    input_text = re.sub(r'No generated caption found', '', input_text)
    input_text = re.sub(r'\s+', ' ', input_text).strip()

    # Prepare input for model
    input_text = f"generate story: {input_text}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    # Generate output
    with torch.no_grad():
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

def read_train_data(filepath):
    """
    Read the training data file and extract input-output pairs.
    """
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Parse input-output pairs
    pairs = []
    pattern = r'input:(.*?)output:(.*?)(?=input:|$)'
    matches = re.findall(pattern, content, re.DOTALL)
    
    for match in matches:
        input_text = match[0].strip()
        output_text = match[1].strip()
        pairs.append((input_text, output_text))
    
    return pairs

def calculate_metrics(reference, candidate):
    """
    Calculate precision, recall, F1 score, and ROUGE metrics.
    """
    # Tokenize texts
    ref_tokens = word_tokenize(reference.lower())
    cand_tokens = word_tokenize(candidate.lower())
    
    # Calculate ROUGE scores
    rouge = Rouge()
    try:
        rouge_scores = rouge.get_scores(candidate, reference)[0]
        rouge_1 = rouge_scores['rouge-1']
        precision = rouge_1['p']
        recall = rouge_1['r']
        f1 = rouge_1['f']
    except:
        precision = recall = f1 = 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'rouge': {
            'rouge-1': rouge_1 if 'rouge_1' in locals() else {'p': 0, 'r': 0, 'f': 0},
            'rouge-2': rouge_scores['rouge-2'] if 'rouge_scores' in locals() else {'p': 0, 'r': 0, 'f': 0},
            'rouge-l': rouge_scores['rouge-l'] if 'rouge_scores' in locals() else {'p': 0, 'r': 0, 'f': 0}
        }
    }

def evaluate_model(train_data_path, limit=None):
    """
    Evaluate the model on the training data and save results to a JSON file.
    
    Args:
        train_data_path: Path to the training data file
        limit: Optional limit to process only the first X pairs
    """
    pairs = read_train_data(train_data_path)
    
    # Apply limit if specified
    if limit and limit > 0 and limit < len(pairs):
        print(f"Limiting evaluation to first {limit} examples out of {len(pairs)} total")
        pairs = pairs[:limit]
    else:
        print(f"Evaluating all {len(pairs)} examples")
    
    results = []
    
    # Calculate total metrics
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    
    # Terminal progress display
    from tqdm import tqdm
    for i, (input_text, reference_story) in enumerate(tqdm(pairs, desc="Evaluating stories")):
        # Generate story
        generated_story = generate_story(input_text, story_model, story_tokenizer, device)
        
        # Calculate metrics
        metrics = calculate_metrics(reference_story, generated_story)
        
        # Add to totals
        total_precision += metrics['precision']
        total_recall += metrics['recall']
        total_f1 += metrics['f1_score']
        
        # Store results
        results.append({
            'input': input_text,
            'reference_story': reference_story,
            'generated_story': generated_story,
            'metrics': {
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score']
            }
        })
        
        # Print detailed progress in terminal
        print(f"\nExample {i+1}/{len(pairs)}:")
        print(f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1_score']:.4f}")
    
    # Calculate averages
    avg_precision = total_precision / len(pairs) if pairs else 0
    avg_recall = total_recall / len(pairs) if pairs else 0
    avg_f1 = total_f1 / len(pairs) if pairs else 0
    
    # Add summary metrics
    summary = {
        'average_metrics': {
            'precision': avg_precision,
            'recall': avg_recall,
            'f1_score': avg_f1
        },
        'examples': results
    }
    
    # Save results to JSON file
    output_filename = "story_model_evaluation.json"
    if limit and limit > 0 and limit < len(pairs):
        output_filename = f"story_model_evaluation_first_{limit}_examples.json"
        
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    return summary

def main():
    # Command-line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Story Generation Model')
    parser.add_argument('--data', type=str, default="train_data.txt", help='Path to training data file')
    parser.add_argument('--limit', type=int, help='Limit to first X examples')
    parser.add_argument('--output', type=str, default="story_model_evaluation.json", help='Output JSON file path')
    
    args = parser.parse_args()
    
    # Ask user for the number of samples to evaluate if no limit is set
    if args.limit is None:
        limit = input("How many samples would you like to evaluate? (Enter a number, or press Enter to evaluate all): ")
        limit = int(limit) if limit.strip().isdigit() else None
    else:
        limit = args.limit
    
    print(f"Evaluating model using {args.data}...")
    if limit:
        print(f"Limiting to first {limit} examples")
    
    # Run evaluation
    summary = evaluate_model(args.data, limit)
    
    # Print evaluation summary
    metrics = summary['average_metrics']
    
    print("\n" + "="*50)
    print("Evaluation Results:")
    print("="*50)
    print(f"Average Precision: {metrics['precision']:.4f}")
    print(f"Average Recall: {metrics['recall']:.4f}")
    print(f"Average F1 Score: {metrics['f1_score']:.4f}")
    print(f"\nResults saved to '{args.output}'")

if __name__ == "__main__":
    main()
