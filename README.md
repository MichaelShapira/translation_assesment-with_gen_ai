# LLM Translation Quality Assessment with Enhanced Analysis

This repository contains a comprehensive Python application for evaluating the translation quality of various Language Models (LLMs) on product reviews. The system generates authentic-sounding product reviews with different sentiments, translates them using multiple models, and provides multi-dimensional evaluation using LLM judges, BLEU scores, and Sentence Transformers for semantic similarity analysis.

## Overview

Machine translation is a critical application for businesses with global reach. This project provides a comprehensive framework to answer important questions about LLM translation capabilities:

- How do different LLM models compare in translation quality across multiple evaluation metrics?
- Which models provide the best balance of quality, speed, and cost efficiency?
- How do models perform across different language pairs and sentiment types?
- What insights can be gained from combining traditional metrics (BLEU) with modern semantic similarity measures?
- How can AI agents help analyze and visualize translation performance data?

## Key Components

### 1. Core Translation System (`lang_translate.py`)
- **Review Generation**: Creates realistic product reviews with positive, neutral, or negative sentiment in English, Tamil, and Chinese
- **Multi-Model Translation**: Translates content using multiple models:
  - Claude 3.5 Haiku
  - Amazon Nova Micro
  - Amazon Nova Lite
- **Target Languages**: Translates to Russian, Hebrew, and German
- **LLM-as-Judge Evaluation**: Uses Claude 3.7 Sonnet to evaluate translations on:
  - Correctness (accuracy of meaning, technical terms, product details)
  - Sentiment preservation
  - Word count ratio analysis
- **Performance Metrics**: Captures latency, token usage, and quality metrics for each translation

### 2. Enhanced Analysis System (`enhanced_with_sentence_transformers.py`)
- **BLEU Score Validation**: Implements cross-lingual BERTScore using XLM-RoBERTa for translation accuracy measurement
- **Sentence Transformers Analysis**: Uses multilingual sentence transformers for semantic similarity evaluation
- **CPU-Optimized Processing**: Configured for laptop performance with CPU-only execution
- **Extended Model Support**: Includes Meta Llama Maverick 17B for comprehensive comparison
- **Language Focus**: Processes English reviews with translation to Russian and Hebrew
- **Comprehensive Output**: Generates detailed CSV with all evaluation metrics

### 3. AI-Powered Analysis Agent (`visualization_agent.py`)
- **Automated Data Analysis**: Uses Amazon Strands agents for comprehensive performance analysis
- **Cost Analysis**: Integrates with AWS billing APIs for exact cost calculations
- **Advanced Visualizations**: Generates 20+ publication-ready charts and analyses
- **Executive Reporting**: Creates comprehensive reports with actionable insights
- **MCP Integration**: Leverages Model Context Protocol servers for data processing

## Key Findings

Based on comprehensive analysis including BLEU scores, Sentence Transformers, and AI agent evaluation:

### 🏆 Meta Llama Maverick 17B (Top Performer)
- **Best Cost Efficiency**: $0.000331 per translation
- **Highest Quality**: 0.906 composite score combining all evaluation metrics
- **Best ROI**: 2,768 performance points per dollar
- **Balanced Performance**: Excellent across correctness (9.0/10), sentiment (9.8/10), BLEU F1 (0.913), and semantic similarity (0.823)

### ⚡ Nova Micra (Speed Optimized)
- **Fastest Response**: 1.80s average latency
- **Highest Throughput**: 224.4 tokens/second
- **Good Cost-Performance**: Competitive pricing with fast processing
- **Ideal for**: High-volume applications requiring quick turnaround

### 🎯 Claude 3.5 Haiku (Quality Focused)
- **Excellent Quality**: 0.904 composite score with highest semantic similarity (0.834)
- **Most Consistent**: Lowest variance across different translation tasks
- **Premium Performance**: Strong correctness (9.0/10) and sentiment preservation (9.8/10)
- **Best for**: Applications requiring highest translation accuracy

### 💼 Nova Lite (Enterprise Option)
- **Reliable Performance**: Consistent quality across different content types
- **Good Speed**: Competitive processing times
- **Enterprise Features**: Suitable for business applications with specific requirements
- **Balanced Approach**: Offers good performance across multiple metrics

## Enhanced Evaluation Metrics

### Traditional Metrics
- **BLEU F1 Scores**: All models achieve >0.91, indicating strong translation accuracy
- **Latency Analysis**: Range from 1.80s (Nova Micra) to 4.77s (Claude 3.5 Haiku)
- **Token Efficiency**: Input/output token ratios for cost optimization

### Advanced Semantic Analysis
- **Sentence Transformer Similarity**: Multilingual semantic similarity scores (0.76-0.84 range)
- **Cross-lingual Validation**: XLM-RoBERTa based BLEU scoring for language pair accuracy
- **Composite Quality Scoring**: Weighted combination of all evaluation dimensions

## Multi-Dimensional Evaluation Framework

### 1. LLM as a Judge
This project employs "LLM as a judge" methodology using Claude 3.7 Sonnet to evaluate translation quality:
- **Consistency**: Uniform evaluation criteria across all translations
- **Expertise**: Advanced multilingual capabilities for nuanced assessment
- **Efficiency**: Automated evaluation replacing manual review processes
- **Qualitative Feedback**: Detailed assessments beyond simple numeric scores

**Evaluation Metrics:**
- **Correctness Score (1-10)**: Accuracy of meaning, preservation of technical details
- **Sentiment Score (1-10)**: Emotional tone maintenance and sentiment preservation
- **Word Count Ratio**: Translation length appropriateness analysis

### 2. BLEU Score Validation (CPU-Optimized)
Implements cross-lingual BERTScore using XLM-RoBERTa for objective translation quality measurement:
- **Cross-lingual Comparison**: Direct semantic comparison between source and target languages
- **F1 Score Focus**: Balanced precision and recall for translation accuracy
- **CPU Processing**: Optimized for laptop performance with 2-thread processing
- **Multilingual Support**: Handles English-Russian and English-Hebrew language pairs

### 3. Sentence Transformers Semantic Analysis
Utilizes multilingual sentence transformers for semantic similarity evaluation:
- **Model**: `paraphrase-multilingual-mpnet-base-v2` for cross-lingual semantic understanding
- **Cosine Similarity**: Measures semantic preservation between source and target texts
- **Embedding-based**: Vector space analysis for meaning preservation assessment
- **CPU Execution**: Configured for local processing without GPU requirements

## Requirements

### Core Dependencies
- Python 3.8+
- AWS SDK for Python (Boto3)
- AWS account with Bedrock access

### Enhanced Analysis Dependencies
```bash
# BLEU and Sentence Transformers (CPU-optimized)
pip install bert-score
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install sentence-transformers
pip install scikit-learn
pip install numpy
```

### AI Agent Dependencies
```bash
# Amazon Strands for AI agents
pip install strands-agents
pip install strands-agents-tools
pip install python-dotenv
```

### AWS Model Access Permissions
- Claude 3.5 Sonnet (for review generation)
- Claude 3.5 Haiku (for translation)
- Claude 3.7 Sonnet (for evaluation)
- Amazon Nova Micro and Nova Lite models
- Meta Llama Maverick 17B (for enhanced analysis)

## MCP Server Integration

The AI agent utilizes two Model Context Protocol (MCP) servers for data processing:

### 1. Pandas MCP Server
- **Repository**: https://github.com/marlonluo2018/pandas-mcp-server
- **Purpose**: Advanced data analysis and manipulation
- **License**: MIT (project files not included to avoid duplication)

### 2. SQLite MCP Server
- **Repository**: https://github.com/modelcontextprotocol/servers-archived/tree/HEAD/src/sqlite
- **Purpose**: Database operations and SQL query execution
- **License**: MIT (project files not included to avoid duplication)

## Output Files

### Generated Data
- `vacuum_cleaner_reviews_evaluated.csv` - Original evaluation results
- `enhanced_with_sentence_transformers_evaluation.csv` - Enhanced analysis with BLEU and ST scores
- `mydatabase.db` - SQLite database for agent analysis

### Analysis Reports (in `output_report/` folder)
- `final_report.md` - Comprehensive analysis report with 20+ visualizations
- `executive_summary.md` - High-level findings and recommendations
- `cost_analysis.csv` - Detailed cost breakdown by model
- `performance_analysis.csv` - Performance metrics comparison
- Various PNG visualization files

## Key Features

### Cost Analysis
- Exact cost calculations using AWS billing APIs
- Cost per translation, cost per word metrics
- ROI analysis and cost-efficiency comparisons
- Volume-based cost projections

### Performance Visualization
- 20+ publication-ready charts and analyses
- Cost vs. quality trade-off visualizations
- Latency and throughput comparisons
- Quality consistency analysis across sentiment categories

### Multi-Language Support
- **Core System**: Source languages (English, Tamil, Chinese) → Target languages (Russian, Hebrew, German)
- **Enhanced Analysis**: Source language (English) → Target languages (Russian, Hebrew)
- **Cross-lingual evaluation capabilities** with BLEU and Sentence Transformers

## Future Enhancements
- Expand to additional language pairs and models
- Implement real-time translation quality monitoring
- Add confidence intervals and statistical significance testing
- Develop web-based dashboard for interactive analysis
- Integration with additional MCP servers for extended functionality

## License
This project is licensed under the MIT License

---
*This README was automatically generated by a generative AI model.* 
