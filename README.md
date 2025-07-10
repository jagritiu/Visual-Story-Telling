
# Image Story Generation with Fine-Tuned BLIP and T5 Transformer Model

This repository contains code for generating stories from a sequence of 5 images using a fine-tuned BLIP model for image captioning and a T5 transformer model for story generation.

## Overview

The goal of this project is to generate a coherent narrative based on a set of 5 images. The workflow involves:
1. Using a fine-tuned BLIP model to generate captions for each image.
2. Feeding the generated captions into a T5 transformer model to generate a story that ties the captions together.

### Files

1. **train_t5.py**  
   This script is used to train the T5 transformer model on a dataset for the story generation task. The model is fine-tuned to take image captions as input and generate a narrative.

2. **generate_story_from_image.py**  
   This script runs the final program. It takes 5 images, generates captions for them using the fine-tuned BLIP model, and then feeds these captions into the trained T5 model to generate a story based on the images.

3. **story_model_evaluation.json**  
   This file contains the evaluation results of the T5 story generation model. It includes:
   - **Average Metrics**: The average Precision, Recall, F1 Score, and ROUGE scores (for individual and aggregated results).
   - **Evaluation Data**: The model’s output (generated story) compared to the reference (ground truth) story for each example in the evaluation set, along with the calculated scores for Precision, Recall, F1, and ROUGE for each evaluation example.
   
   This file is generated after evaluating the model on a set of test data, providing detailed insights into how well the model performs in generating stories. The file is in JSON format and contains the following structure:
   - `average_metrics`: A dictionary containing the average Precision, Recall, F1 Score, and ROUGE metrics.
   - `examples`: A list of examples with the input captions, reference stories, generated stories, and the metrics for each evaluation example.

---
