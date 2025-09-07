import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

def load_and_process_data():
    """Load and process the translation benchmarking data"""
    
    # Load the CSV data
    df = pd.read_csv('/Users/michshap/Documents/Code/translation_assesment-with_gen_ai/enhanced_with_sentence_transformers_evaluation.csv')
    
    # Restructure the data into a long format for easier analysis
    models = ['Claude 3.5 Haiku', 'Nova Micra', 'Nova Lite', 'Meta llama maverick 17B']
    
    # Create a list to store restructured data
    restructured_data = []
    
    for idx, row in df.iterrows():
        sentiment = row['Prompt sentiment to generate the review']
        original_text = row['Result']
        
        for model in models:
            # Extract metrics for each model
            translation = row[model]
            input_tokens = int(row[f'{model} input tokens'])
            output_tokens = int(row[f'{model} output tokens'])
            latency = float(row[f'{model} latency'])
            correctness_score = int(row[f'{model} correctness score'])
            sentiment_score = int(row[f'{model} sentiment score'])
            word_count_ratio = float(row[f'{model} word count ratio'])
            bleu_f1_score = float(row[f'{model} BLEU F1 score'])
            st_similarity_score = float(row[f'{model} ST similarity score'])
            overall_assessment = row[f'{model} overall assessment']
            
            # Determine target language based on translation content
            if any('\u0400' <= char <= '\u04FF' for char in translation):
                target_language = 'Russian'
            elif any('\u0590' <= char <= '\u05FF' for char in translation):
                target_language = 'Hebrew'
            else:
                target_language = 'Unknown'
            
            restructured_data.append({
                'row_id': idx,
                'sentiment_category': sentiment,
                'original_text': original_text,
                'model': model,
                'target_language': target_language,
                'translation': translation,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'total_tokens': input_tokens + output_tokens,
                'latency': latency,
                'correctness_score': correctness_score,
                'sentiment_score': sentiment_score,
                'word_count_ratio': word_count_ratio,
                'bleu_f1_score': bleu_f1_score,
                'st_similarity_score': st_similarity_score,
                'overall_assessment': overall_assessment
            })
    
    # Create DataFrame from restructured data
    analysis_df = pd.DataFrame(restructured_data)
    
    # Add cost calculations
    pricing_data = {
        'Claude 3.5 Haiku': {'input': 0.25, 'output': 1.25},
        'Nova Micra': {'input': 0.35, 'output': 1.40},
        'Nova Lite': {'input': 0.60, 'output': 2.40},
        'Meta llama maverick 17B': {'input': 0.40, 'output': 1.60}
    }
    
    # Calculate costs
    analysis_df['input_cost'] = analysis_df.apply(
        lambda row: (row['input_tokens'] / 1_000_000) * pricing_data[row['model']]['input'], axis=1
    )
    analysis_df['output_cost'] = analysis_df.apply(
        lambda row: (row['output_tokens'] / 1_000_000) * pricing_data[row['model']]['output'], axis=1
    )
    analysis_df['total_cost'] = analysis_df['input_cost'] + analysis_df['output_cost']
    
    # Calculate additional cost metrics
    analysis_df['cost_per_word'] = analysis_df['total_cost'] / analysis_df['original_text'].str.split().str.len()
    analysis_df['cost_per_quality_point'] = analysis_df['total_cost'] / analysis_df['correctness_score']
    
    return analysis_df

def generate_cost_analysis(df):
    """Generate comprehensive cost analysis"""
    
    # Cost summary by model
    cost_summary = df.groupby('model').agg({
        'total_cost': ['mean', 'sum', 'min', 'max', 'std'],
        'input_cost': 'mean',
        'output_cost': 'mean',
        'cost_per_word': 'mean',
        'cost_per_quality_point': 'mean',
        'input_tokens': 'mean',
        'output_tokens': 'mean'
    }).round(6)
    
    return cost_summary

def generate_performance_analysis(df):
    """Generate comprehensive performance analysis"""
    
    # Performance summary by model
    performance_summary = df.groupby('model').agg({
        'correctness_score': ['mean', 'std', 'min', 'max'],
        'sentiment_score': ['mean', 'std', 'min', 'max'],
        'bleu_f1_score': ['mean', 'std', 'min', 'max'],
        'st_similarity_score': ['mean', 'std', 'min', 'max'],
        'latency': ['mean', 'std', 'min', 'max'],
        'word_count_ratio': ['mean', 'std', 'min', 'max']
    }).round(4)
    
    return performance_summary

if __name__ == "__main__":
    # Load and process data
    analysis_df = load_and_process_data()
    
    # Generate analyses
    cost_analysis = generate_cost_analysis(analysis_df)
    performance_analysis = generate_performance_analysis(analysis_df)
    
    # Save processed data
    analysis_df.to_csv('output_report/processed_translation_data.csv', index=False)
    
    # Save analysis results
    cost_analysis.to_csv('output_report/cost_analysis.csv')
    performance_analysis.to_csv('output_report/performance_analysis.csv')
    
    print("Analysis completed and saved!")
    print(f"Total records processed: {len(analysis_df)}")
    print(f"Models analyzed: {analysis_df['model'].nunique()}")
    print(f"Languages: {analysis_df['target_language'].unique()}")