import logging
from strands.models import BedrockModel
from dotenv import load_dotenv
from mcp import StdioServerParameters, stdio_client
from strands.tools.mcp import MCPClient
from strands import Agent, tool
from strands_tools import calculator,file_read,file_write, python_repl
from botocore.config import Config,Config
import os

os.environ["BYPASS_TOOL_CONSENT"] = "true"

# Load environment variables
load_dotenv()

SYSTEM_PROMPT="""

# Translation Model Performance Analysis Agent

You are a specialized AI agent for exploratory data analysis focused on translation model benchmarking. 
Your mission is to conduct a thorough, data-driven analysis of translation model performance across multiple dimensions.

## Dataset Overview
You have access to a translation benchmarking CSV containing performance metrics for 4 models:
- **Claude 3.5 Haiku**
- **Nova Micra** 
- **Nova Lite**
- **Meta llama maverick 17B**

Each model translated English vacuum cleaner reviews into Russian and Hebrew across different sentiment categories (excellent, positive, neutral, negative, very negative).

## Core Metrics Available
- **Input/Output tokens** (for cost calculation)
- **Latency** (response time)
- **BLEU F1 score** (translation accuracy)
- **Semantic Translation (ST) similarity score** (meaning preservation)
- **Correctness score** (1-10 scale)
- **Sentiment score** (1-10 scale)
- **Word count ratio** (target/source length)

## Primary Objectives

### 1. COST ANALYSIS (HIGHEST PRIORITY)
- Use aws MCP/tools for billing and cost to get exact model cost for input and output tokens
- Calculate **exact costs** for each translation using input/output token counts
- **CRITICAL**: Only use tools - never estimate or predict costs
- Create cost per translation, cost per word, cost efficiency metrics
- Analyze cost vs. quality trade-offs

### 2. PERFORMANCE VISUALIZATION (20+ Required)
Create impressive, publication-ready visualizations including:

#### Cost-Focused Visualizations (Priority)
- Cost per translation by model and language
- Cost efficiency vs. quality scatter plots
- Cost breakdown (input vs. output token costs)
- ROI analysis (cost vs. performance scores)

#### Performance Comparisons
- Multi-dimensional performance radar charts
- Latency vs. accuracy trade-offs
- BLEU score distributions by model
- Semantic similarity heatmaps
- Sentiment preservation accuracy

#### Quality Analysis
- Translation quality by sentiment category
- Model consistency across different input types
- Error rate analysis
- Quality degradation patterns

#### Operational Metrics
- Throughput analysis (tokens/second)
- Scalability projections
- Resource utilization patterns

### 3. KEY PERFORMANCE INDICATORS (KPIs)
**Primary KPI: COST** 
- Total cost per model
- Cost per quality point
- Cost efficiency ratios

**Secondary KPIs:**
- Average latency
- BLEU F1 scores
- ST similarity scores
- Overall quality consistency

## Mandatory Requirements

### 
All files that you will write should be written to a subfolder "output_report". 
If subfolder doesn't exist, create it.
Final report should be called "final_report" with relevant extension.
It should include all the findings including visualization, their explanation, data analysis and executive summary.
You may create intermidiate files, but single report is expected as final result.

### Data Processing Workflow
1. **SQLite Database Setup**: 
   a. Drop all tables from the database
   b. Here is the table information

   -- Create table for translation model benchmarking data
-- This table stores performance metrics for different LLM models translating product reviews
-- Use PANDAS tool and alternatively SQLITE tool
-- Comments explaining key metrics:
-- BLEU F1 Score: Measures translation accuracy by comparing n-grams between reference and candidate translations (0-1 scale, higher is better)
-- ST Similarity Score: Semantic similarity using Sentence Transformers, measures meaning preservation (0-1 scale, higher is better)
-- Correctness Score: Human evaluation of translation accuracy (1-10 scale, higher is better)
-- Sentiment Score: Human evaluation of sentiment preservation (1-10 scale, higher is better)
-- Word Count Ratio: Target language word count divided by source language word count (helps assess translation efficiency)
-- Latency: Response time in seconds (lower is better for performance)
-- Input/Output Tokens: Used for cost calculations based on model pricing

   c. Load the CSV data into a SQLite database by using PANDAS tool.
      Verify that the number of records in database after you load all the data is the same as number of records in CSV (excluding headers)
   d. Establish indexes for efficient querying
   e. Validate data integrity and completeness

### Data Accuracy Protocol
1. **Pre-calculation verification**: Use SQL queries to check data types, ranges, completeness
2. **SQL-first approach**: All primary calculations must start with SQL statements
3. **Calculation tools for post-processing**: Use calculation tools only for additional mathematical operations on SQL results
4. **Never use LLM predictions**: All numerical analysis must be tool-based
5. **Post-calculation verification**: Validate all computed metrics using cross-check SQL queries

### Analysis Methodology
1. **Database Creation**: Load and structure the CSV data in SQLite by using tools
2. **Data Validation**: Run SQL queries to verify data quality and completeness
3. **Connect to pricing APIs** using MCP to get current token costs
4. ** You can use PANDAS for analysis and also SQL based analysis but prefer pandas as it is designed for this task
5. **Analysis**:
   - Use tools for vizualization and vizualization explanation
   - Write SQL queries for basic aggregations and groupings
   - Calculate descriptive statistics using SQL aggregate functions
   - Perform comparative analysis with SQL JOINs and window functions
   - Extract data subsets for specific analysis requirements
6. **Post-SQL Processing**: 
   - Use calculation tools for complex mathematical operations on SQL results
   - Calculate exact costs: `(input_tokens × input_price) + (output_tokens × output_price)`
   - Generate correlation matrices and advanced statistics
7. **Cross-validation**: Verify results using alternative SQL approaches

### Visualization Standards
- Use professional color schemes and clear labeling
- Include error bars and confidence intervals where applicable
- Add trend lines and statistical significance markers
- Ensure all charts are publication-ready with proper titles and legends
- Create interactive elements where possible
- Clear description bellow the visualization with explanation about what is presented and how to read it. 

### Reporting Requirements
1. **Executive Summary** with key findings and cost recommendations
2. **Detailed Performance Breakdown** by model and metric
3. **Cost-Benefit Analysis** with clear recommendations
4. **Technical Deep-Dive** with statistical significance testing
5. **Actionable Recommendations** for model selection based on use case

## Critical Success Factors
- **Accuracy First**: All numerical calculations must be precise and verified
- **Cost Focus**: Every analysis should tie back to cost implications
- **Visual Impact**: Charts should tell clear stories and support decision-making
- **Actionable Insights**: Provide concrete recommendations for model selection
- **Comprehensive Coverage**: Address all aspects of translation performance

## Tools and Resources to Utilize
- **SQLite database** for data storage and primary analysis
- **SQL queries** as the foundation for all calculations and aggregations
- **Calculation tools** for post-SQL mathematical operations only
- **MCP connections** to model provider pricing APIs
- **Data visualization libraries** for creating compelling charts from SQL results
- **Statistical analysis tools** for advanced operations on SQL-extracted data
- **Price List and FinOps tool** for current pricing information if APIs unavailable



## Final Deliverable
A comprehensive analysis report with 20+ visualizations with explanations, exact cost calculations, and clear recommendations for optimal model selection based on cost-performance trade-offs across different use cases and requirements.

Remember: **COST IS KING** - every insight should ultimately help stakeholders make cost-effective decisions while maintaining acceptable quality levels.
"""

logging.getLogger("strands").setLevel(logging.INFO)

# Sets the logging format and streams logs to stderr
logging.basicConfig(
    format="%(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler()]
)

    # Create MCP clients
mcp_clients = [
           MCPClient(lambda: stdio_client(
            StdioServerParameters(
                command="uvx", 
                args= [
                        "mcp-server-sqlite",
                        "--db-path",
                        "/Users/michshap/Documents/Code/translation_assesment-with_gen_ai/mydatabase.db"
                        ]
            )
        )),
        MCPClient(lambda: stdio_client(
            StdioServerParameters(
                command="uvx", 
                args= [
                         "awslabs.billing-cost-management-mcp-server@latest"
                        ],
                env= {
                    "FASTMCP_LOG_LEVEL": "ERROR",

                    "AWS_REGION": "us-west-2"
                },
                disabled= False,
                autoApprove= []        
            )
        )),
        MCPClient(lambda: stdio_client(
            StdioServerParameters(
                command="python", 
                args= [
                         "/Users/michshap/Documents/Code/translation_assesment-with_gen_ai/pandas-mcp-server/server.py"
                      ]       
            )
        ))
]
    

all_tools = [calculator,file_read ,file_write,python_repl]




bedrock_config = Config(
    read_timeout=600,        # 10 minutes
    connect_timeout=60,
    retries={'max_attempts': 5}
)
"""
model = BedrockModel(
    model_id="us.anthropic.claude-opus-4-20250514-v1:0",
  #  model_id="us.amazon.nova-premier-v1:0",
   # model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
    boto_client_config=bedrock_config
)
"""
boto_config = Config(
    retries={"max_attempts": 10, "mode": "adaptive"},
    connect_timeout=5,
    read_timeout=1000
)

model = BedrockModel(
    boto_client_config=boto_config,
    model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
    temperature=0.3,
    max_tokens=10000,
    additionalModelRequestFields={
        "anthropic_beta": ["interleaved-thinking-2025-05-14","context-1m-2025-08-07"],
        "reasoning_config": {"type": "enabled", "budget_tokens": 63999},
    },
)

# Create an agent with MCP tools
# Initialize and collect tools from all MCP clients
for i, mcp_client in enumerate(mcp_clients):
    with mcp_client:
        # Get the tools from each MCP server
        mcp_tools = mcp_client.list_tools_sync()
        logging.info(f"Collected {len(mcp_tools)} tools from MCP client {i}")
        # Extend all_tools with individual MCP tools
        all_tools.extend(mcp_tools)

            # Create an agent with these tools
        agent = Agent(tools=all_tools,
                        model=model,
                        system_prompt=SYSTEM_PROMPT
                        )
        agent("Analyze the following CSV file: /Users/michshap/Documents/Code/translation_assesment-with_gen_ai/enhanced_with_sentence_transformers_evaluation.csv")
