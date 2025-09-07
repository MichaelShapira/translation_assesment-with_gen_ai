import boto3
import json
import time
import csv
import logging
import re
from typing import List, Dict, Any, Tuple, Optional

# BLEU validation imports
from bert_score import score
import torch
import os

# Sentence Transformers imports
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Force CPU usage and optimize for laptop performance
torch.set_num_threads(2)
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Initialize clients (will be initialized when needed)
bedrock_client = None
sentence_transformer_model = None

def get_bedrock_client():
    """Initialize bedrock client when needed"""
    global bedrock_client
    if bedrock_client is None:
        try:
            bedrock_client = boto3.client(
                service_name="bedrock-runtime",
                region_name="us-west-2"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Bedrock client: {str(e)}")
            raise
    return bedrock_client

def get_sentence_transformer_model():
    """Initialize Sentence Transformer model when needed"""
    global sentence_transformer_model
    if sentence_transformer_model is None:
        try:
            logger.info("Loading Sentence Transformer model (first run may take a few minutes)...")
            sentence_transformer_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
            logger.info("Sentence Transformer model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Sentence Transformer model: {str(e)}")
            raise
    return sentence_transformer_model

def evaluate_cross_lingual_translation(source_text: str, target_translation: str, 
                                     source_lang: str = "en", target_lang: str = "ru") -> float:
    """
    Evaluate translation quality using cross-lingual BERTScore with XLM-RoBERTa
    Returns only F1 score as requested
    """
    try:
        logger.info(f"Running BLEU validation for {source_lang} → {target_lang}")
        
        # Use XLM-RoBERTa for cross-lingual comparison
        P, R, F1 = score(
            cands=[target_translation], 
            refs=[source_text], 
            model_type="xlm-roberta-large",
            verbose=False  # Reduce output noise
        )
        
        # Return only F1 score as requested
        return F1.item()
        
    except Exception as e:
        logger.error(f"Error in BLEU validation: {str(e)}")
        return 0.0

def evaluate_with_sentence_transformers(source_text: str, target_translation: str) -> float:
    """
    Evaluate translation quality using Sentence Transformers
    Returns only cosine similarity score as requested
    """
    try:
        logger.info("Running Sentence Transformer evaluation...")
        
        model = get_sentence_transformer_model()
        
        # Generate embeddings for both texts
        embeddings = model.encode([
            source_text.strip(),
            target_translation.strip()
        ], 
        show_progress_bar=False,
        convert_to_tensor=False
        )
        
        # Calculate cosine similarity
        source_embedding = embeddings[0].reshape(1, -1)
        target_embedding = embeddings[1].reshape(1, -1)
        
        similarity_matrix = cosine_similarity(source_embedding, target_embedding)
        similarity_score = similarity_matrix[0][0]
        
        return float(similarity_score)
        
    except Exception as e:
        logger.error(f"Error in Sentence Transformer validation: {str(e)}")
        return 0.0

def generate_review(language: str, sentiment: str) -> str:
    """Generate a vacuum cleaner review in the specified language with the given sentiment."""
    stars = {"excelent":5,"positive": 4, "neutral": 3, "negative": 2,"very negative":1}
    
    # Create the prompt for generating the review
    prompt = (
        f"Write the review in {language} language about imaginary vacuum cleaner. "
        f"Make it around 100 words. The review should reflect {stars[sentiment]} star rating "
        "(1 is the lowest, 5 is the highest). "
        "Deliberately make several grammar mistakes. "
        f"Make the sentiment {sentiment}. Keep the language simple. Write review only, without title"
    )
    
    # Use Claude to generate the review
    client = get_bedrock_client()
    response = client.converse(
        modelId="us.anthropic.claude-opus-4-1-20250805-v1:0",
        messages=[
            {
                "role": "user",
                "content": [{"text": prompt}],
            }
        ],
        system=[{"text": "You are helpful AI assistant that creates authentic-sounding product reviews."}],
        inferenceConfig={"temperature": 0.7, "maxTokens": 500}
    )
    
    return prompt, response["output"]["message"]["content"]

def translate_review(review: str, model_id: str, target_lang: str) -> Dict:
    """Translate a review using the specified model and return results with metrics."""
    system_prompt = [{"text":f"Translate to {target_lang}. Keep the sentiment. Output translated text only."}]
    
    messages = [
                {"role": "user", "content": [{"text": review}]},
    ]
    inference_config = {
        "temperature": 0,
        "maxTokens": 500
    }
    
    additional_model_fields = {}
    
    # Start timer
    start_time = time.time()
    
    # Send the message.
    client = get_bedrock_client()
    response = client.converse(
        modelId=model_id,
        messages=messages,
        system=system_prompt,
        inferenceConfig=inference_config,
        additionalModelRequestFields=additional_model_fields
    )
    
    # Calculate latency
    latency = time.time() - start_time
    
    # Get translated text
    translated_text = response["output"]["message"]["content"]
    
    # Log token usage
    token_usage = response['usage']
    logger.info("Input tokens: %s", token_usage['inputTokens'])
    logger.info("Output tokens: %s", token_usage['outputTokens'])
    logger.info("Total tokens: %s", token_usage['totalTokens'])
    logger.info("Stop reason: %s", response['stopReason'])
    logger.info("Request latency: %.2f seconds", latency)
    
    return {
        "translated_text": translated_text,
        "input_tokens": token_usage['inputTokens'],
        "output_tokens": token_usage['outputTokens'],
        "latency": latency,
        "model_id": model_id
    }

def evaluate_translation(original_text: str, translated_text: str, source_lang: str, target_lang: str) -> Dict:
    """Evaluate translation quality using Claude 4 ."""
    
    evaluation_prompt = f"""
You are an expert in translation quality assessment specializing in product reviews. You will evaluate translations from {source_lang} to {target_lang} by analyzing the following key metrics:

1. CORRECTNESS (Score 1-10): Assess how accurately the translation preserves the original meaning, technical terms, and product details. Consider:
   - Are all key points from the original text included?
   - Are there mistranslations or omissions of important information?
   - Are product-specific terms translated properly?

2. SENTIMENT PRESERVATION (Score 1-10): Evaluate how well the translation maintains the emotional tone and customer sentiment of the original review:
   - Does the translation convey the same level of satisfaction/dissatisfaction?
   - Are expressions of emotion (excitement, disappointment, etc.) preserved?
   - Does the translation maintain the review's overall recommendation stance?

3. LENGTH ANALYSIS:
   - Calculate word count ratio (translation word count / original word count)
   - Ideal ratio often varies by language pair - note if the ratio seems appropriate

Analyze these translation pairs and respond with ONLY the following values in a JSON format:
- correctness_score: Numeric score 1-10
- sentiment_score: Numeric score 1-10
- original_word_count: Number of words in original
- translation_word_count: Number of words in translation
- word_count_ratio: Calculated ratio
- overall_assessment: Brief evaluation (1-2 sentences)

Original ({source_lang}): {original_text}
Translation ({target_lang}): {translated_text}

Output the evaluation results in valid JSON format only.
"""
    try:
        client = get_bedrock_client()
        response = client.converse(
            modelId="us.anthropic.claude-opus-4-1-20250805-v1:0",
            messages=[
                {
                    "role": "user",
                    "content": [{"text": evaluation_prompt}],
                }
            ],
            system=[{"text": "You are a translation quality evaluator that outputs only valid JSON."}],
            inferenceConfig={"temperature": 0, "maxTokens": 1000}
        )
        
        response_text = response["output"]["message"]["content"][0]["text"]
        # Extract JSON from response if it's embedded in other text
        json_match = re.search(r'({[\s\S]*})', response_text)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response_text
            
        # Parse the JSON results
        evaluation_results = json.loads(json_str)
        return evaluation_results
        
    except Exception as e:
        logger.error(f"Error in translation evaluation: {str(e)}")
        return {
            "correctness_score": 0,
            "sentiment_score": 0,
            "original_word_count": 0,
            "translation_word_count": 0,
            "word_count_ratio": 0,
            "overall_assessment": f"Evaluation failed: {str(e)}"
        }

def get_model_display_name(model_id: str) -> str:
    """Get a display name for a model ID."""
    if "claude" in model_id.lower():
        return "Claude 3.5 Haiku"
    elif "nova-micro" in model_id.lower():
        return "Nova Micra"
    elif "nova-lite" in model_id.lower():
        return "Nova Lite"
    elif "llama" in model_id.lower():
        return "Meta llama maverick 17B"
    else:
        return model_id.split(":")[-2].split("-")[-1]

def run_enhanced_with_sentence_transformers():
    """Main function that runs the enhanced translation evaluation with BLEU and Sentence Transformers validation."""
    # Configuration
    source_languages = ["English"]
    target_languages = ["Russian","Hebrew"]
    target_languages_code = {"Russian":"ru","Hebrew": "he"}
    model_ids = [
        "us.anthropic.claude-3-5-haiku-20241022-v1:0",
        "us.amazon.nova-micro-v1:0",
        "us.amazon.nova-lite-v1:0",
        "us.meta.llama4-maverick-17b-instruct-v1:0"
    ]
    
    # Prepare the CSV file
    csv_file = "enhanced_with_sentence_transformers_evaluation.csv"
    
    # Define base CSV headers
    base_headers = [
        "Prompt sentiment to generate the review", "Result"
    ]
    
    # Create extended headers with evaluation metrics for each model
    headers = base_headers.copy()
    model_display_names = [get_model_display_name(model_id) for model_id in model_ids]
    
    for model_name in model_display_names:
        # Add translation output and metrics columns (including BLEU F1 and Sentence Transformer scores)
        headers.extend([
            f"{model_name}",  # Translation text
            f"{model_name} input tokens", 
            f"{model_name} output tokens", 
            f"{model_name} latency",
            # Add evaluation metrics columns
            f"{model_name} correctness score",
            f"{model_name} sentiment score",
            f"{model_name} word count ratio",
            f"{model_name} overall assessment",
            f"{model_name} BLEU F1 score",  # BLEU validation column
            f"{model_name} ST similarity score"  # NEW: Sentence Transformer validation column
        ])
    
    with open(csv_file, 'w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        
        sentiments = ["excelent","positive", "neutral", "negative","very negative"]
        
        # For each language, generate reviews with different sentiments
        for language in source_languages:
            for sentiment in sentiments:
                # Generate the review
                prompt, review = generate_review(language, sentiment)
                
                # For each target language, translate with different models
                for target_lang in target_languages:
                    row_data = [sentiment, review[0]['text']]
                    
                    original_text = review[0]['text']
                    
                    # For each model, translate, evaluate, and record metrics
                    for i, model_id in enumerate(model_ids):
                        model_name = model_display_names[i]
                        logger.info(f"Processing with {model_name} from {language} to {target_lang}")
                        
                        try:
                            # Translate the review
                            result = translate_review(original_text, model_id, target_lang)
                            translated_text = result["translated_text"][0]["text"]
                            
                            # Add translation and metrics to row
                            row_data.extend([
                                translated_text,
                                result["input_tokens"],
                                result["output_tokens"],
                                result["latency"]
                            ])
                            
                            # Evaluate the translation using Claude as judge
                            logger.info(f"Evaluating translation quality for {model_name}")
                            evaluation = evaluate_translation(
                                original_text=original_text,
                                translated_text=translated_text,
                                source_lang=language,
                                target_lang=target_lang
                            )
                            
                            # Add evaluation metrics to row
                            row_data.extend([
                                evaluation["correctness_score"],
                                evaluation["sentiment_score"],
                                evaluation["word_count_ratio"],
                                evaluation["overall_assessment"]
                            ])
                            
                            # Add BLEU validation F1 score
                            logger.info(f"Running BLEU validation for {model_name}")
                            f1_score = evaluate_cross_lingual_translation(
                                source_text=original_text,
                                target_translation=translated_text,
                                source_lang="en",
                                target_lang=target_languages_code[target_lang]
                            )
                            row_data.append(f1_score)
                            
                            # NEW: Add Sentence Transformer validation score
                            logger.info(f"Running Sentence Transformer validation for {model_name}")
                            st_score = evaluate_with_sentence_transformers(
                                source_text=original_text,
                                target_translation=translated_text
                            )
                            row_data.append(st_score)
                            
                            logger.info(f"Successfully processed {model_name} - F1: {f1_score:.4f}, ST: {st_score:.4f}")
                            
                        except Exception as e:
                            logger.error(f"Error processing with {model_name}: {str(e)}")
                            # Add placeholder data for translation, evaluation, BLEU, and ST on error
                            row_data.extend(["ERROR", 0, 0, 0, 0, 0, 0, "Error in processing", 0.0, 0.0])
                    
                    # Write the completed row to CSV
                    writer.writerow(row_data)
    
    logger.info(f"Enhanced evaluation with Sentence Transformers complete. Results saved to {csv_file}")

if __name__ == "__main__":
    run_enhanced_with_sentence_transformers()