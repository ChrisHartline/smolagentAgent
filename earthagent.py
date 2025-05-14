from smolagents import (
    CodeAgent, 
    tool,
    DuckDuckGoSearchTool,
    VisitWebpageTool
)
from smolagents.models import LiteLLMModel
from smolagents.prompts import CODE_SYSTEM_PROMPT
import os
from huggingface_hub import login
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import requests
import json

# Load environment variables and authenticate
load_dotenv()
hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
if hf_token:
    login(token=hf_token)
    print("Successfully logged in to Hugging Face Hub")
else:
    print("No Hugging Face API token found. Some functionalities may be limited.")

api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY environment variable is not set")

# Commented out Wolfram Alpha API key check
# wolfram_api_key = os.environ.get("WOLFRAM_ALPHA_APPID")
# if not wolfram_api_key:
#     print("WOLFRAM_ALPHA_APPID not found in environment variables. Wolfram Alpha tool will not work properly.")

# Memory System
class AgentMemory:
    def __init__(self):
        self.short_term = []
        self.summary = ""
        self.max_short_term = 10
    
    def add_interaction(self, step_number, thought, action, observation):
        self.short_term.append({
            "step": step_number,
            "thought": thought,
            "action": action,
            "observation": observation
        })
        
        if len(self.short_term) > self.max_short_term:
            self.short_term.pop(0)
    
    def update_summary(self, new_summary):
        self.summary = new_summary
    
    def get_context(self):
        context = "SUMMARY OF PREVIOUS STEPS:\n" + self.summary + "\n\n"
        context += "RECENT STEPS:\n"
        
        for item in self.short_term:
            context += f"Step {item['step']}:\n"
            context += f"Thought: {item['thought']}\n"
            context += f"Action: {item['action']}\n"
            context += f"Observation: {item['observation']}\n\n"
        
        return context

# Create a global memory instance
agent_memory = AgentMemory()
current_step = 0

# ============================================ Custom Tool Definitions ============================================ #

@tool 
def read_pdf_file(file_path: str) -> str:
    """
    This function reads a PDF file and returns its content as a string.
    
    Args:
        file_path: The path to the PDF file.
    
    Returns:
        A string containing the content of the PDF file.
    """
    try:
        from pypdf import PdfReader
        
        content = ""
        reader = PdfReader(file_path)
        print(f"PDF has {len(reader.pages)} pages")
        
        # Read all pages (or limit to a reasonable number for large PDFs)
        max_pages = min(50, len(reader.pages))  # Limit to 50 pages for very large PDFs
        for i in range(max_pages):
            content += reader.pages[i].extract_text() + "\n"
            
        return content
    except Exception as e:
        return f"Error processing PDF: {str(e)}"

@tool
def get_hugging_face_top_daily_paper() -> str:
    """
    This is a tool that returns the most upvoted paper on Hugging Face daily papers.
    It returns the title of the paper.
    """
    try:
        url = "https://huggingface.co/papers"
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")

        containers = soup.find_all('div', class_='SVELTE_HYDRATER contents')
        top_paper = ""

        for container in containers:
            data_props = container.get('data-props', '')
            if data_props:
                try:
                    json_data = json.loads(data_props.replace('&quot;', '"'))
                    if 'dailyPapers' in json_data:
                        top_paper = json_data['dailyPapers'][0]['title']
                except json.JSONDecodeError:
                    continue

        return top_paper
    except requests.exceptions.RequestException as e:
        print(f"Error occurred while fetching the HTML: {e}")
        return "Failed to fetch top paper due to an error."

# Commented out Wolfram Alpha tool - Using built-in math capabilities instead
# @tool
# def wolfram_alpha(query: str) -> str:
#     """
#     Makes a query to Wolfram Alpha and returns the result.
#     USE THIS TOOL FOR: Mathematical calculations, equations, scientific data, 
#     unit conversions, or any factual query requiring precise numerical answers.
#     
#     Args:
#         query: The query to send to Wolfram Alpha. This can be a mathematical expression,
#                scientific question, or other factual query that Wolfram Alpha can answer.
#     
#     Returns:
#         The result from Wolfram Alpha as text.
#     """
#     try:
#         # Base URL for Wolfram Alpha Simple API
#         base_url = "https://api.wolframalpha.com/v1/result"
#         
#         # Parameters for the API request
#         params = {
#             "appid": wolfram_api_key,
#             "i": query,
#             "units": "metric"  # Use metric units by default
#         }
#         
#         # Make the request
#         response = requests.get(base_url, params=params)
#         
#         # Check if the request was successful
#         if response.status_code == 200:
#             return response.text
#         elif response.status_code == 501:
#             return "Wolfram Alpha couldn't understand the query or has no answer."
#         else:
#             return f"Error: Wolfram Alpha API returned status code {response.status_code}"
#             
#     except Exception as e:
#         return f"Error querying Wolfram Alpha: {str(e)}"

@tool
def math_calculation(expression: str) -> str:
    """
    Evaluates a mathematical expression using Python's built-in math capabilities.
    USE THIS TOOL FOR: Basic mathematical calculations, equations, and evaluations.
    
    Args:
        expression: A mathematical expression as a string (e.g. "2 + 2", "sin(30)", "sqrt(16)")
    
    Returns:
        The result of the calculation as a string
    """
    try:
        import math
        
        # Create a safe dictionary of allowed math functions
        safe_dict = {
            'sqrt': math.sqrt,
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'pi': math.pi,
            'e': math.e,
            'abs': abs,
            'pow': pow,
            'round': round,
            'floor': math.floor,
            'ceil': math.ceil,
            'log': math.log,
            'log10': math.log10,
            'degrees': math.degrees,
            'radians': math.radians
        }
        
        # Add all math module functions
        for name, func in math.__dict__.items():
            if callable(func) and not name.startswith('_'):
                safe_dict[name] = func
        
        # Evaluate the expression using the safe dictionary
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        
        # Return the result as a string
        return str(result)
    except Exception as e:
        return f"Error calculating: {str(e)}"

@tool
def memory_tool(operation: str, content: str = "") -> str:
    """
    Interact with the agent's memory system.
    
    Args:
        operation: The operation to perform (recall, summarize, store)
        content: Content for store operations or query for recall
    
    Returns:
        Result of the memory operation
    """
    global agent_memory
    
    if operation == "recall":
        if content.lower() == "all":
            return agent_memory.get_context()
        elif content.lower() == "summary":
            return f"Current summary: {agent_memory.summary}"
        else:
            matches = []
            for item in agent_memory.short_term:
                if content.lower() in str(item).lower():
                    matches.append(item)
            
            if matches:
                result = "Relevant memories:\n"
                for match in matches:
                    result += f"Step {match['step']}: {match['thought'][:100]}...\n"
                return result
            else:
                return "No relevant memories found."
    
    elif operation == "summarize":
        agent_memory.update_summary(content)
        return f"Summary updated to: {content}"
    
    else:
        return f"Unknown memory operation: {operation}"

@tool
def process_csv(file_path: str, operation: str = "summary") -> str:
    """
    Process CSV data files for various operations.
    USE THIS TOOL FOR: Analyzing tabular data, statistics, or any CSV file processing.
    
    Args:
        file_path: Path to the CSV file
        operation: The operation to perform (summary, count, mean, etc.)
    
    Returns:
        Results of the requested operation
    """
    try:
        import pandas as pd
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        if operation == "summary":
            return str(df.describe())
        elif operation == "columns":
            return str(df.columns.tolist())
        elif operation == "count":
            return str(len(df))
        elif operation == "head":
            return str(df.head())
        elif operation.startswith("mean:"):
            column = operation.split(":", 1)[1].strip()
            return str(df[column].mean())
        else:
            return f"CSV loaded with {len(df)} rows and {len(df.columns)} columns"
    except Exception as e:
        return f"Error processing CSV: {str(e)}"

# Enhanced system prompt with all tool guidelines, updated to use math_calculation instead of wolfram_alpha
enhanced_prompt = CODE_SYSTEM_PROMPT + """
IMPORTANT: Tool Selection and Memory Guidelines

When solving problems, carefully choose the most appropriate tool for each step and use memory to track information:

1. Use memory_tool to:
   - Recall previous information with memory_tool("recall", "query")
   - Summarize important findings with memory_tool("summarize", "your summary")

2. For mathematical calculations, equations, or scientific facts:
   - Use the math_calculation tool for calculations and evaluations
   - Examples: math_calculation("2 + 2"), math_calculation("sin(pi/4)"), math_calculation("sqrt(16)")

3. For web information:
   - Use DuckDuckGoSearchTool for general web searches
   - Use VisitWebpageTool when you need to view the content of a specific URL

4. For file operations:
   - Use read_pdf_file for PDF documents
   - Use process_csv for CSV files and data analysis

5. For research:
   - Use get_hugging_face_top_daily_paper to find the top paper on Hugging Face

When you have completed the task, provide your final answer using:

final_answer("Your complete answer here")

Make sure your final answer is:
1. Directly addresses the original question
2. Is in the exact format requested (if specified)
3. Is concise and clear
4. Only includes the answer itself, not your reasoning (unless requested)
"""

# Memory-enhanced run function
def run_with_memory(agent, query, reset=True):
    global current_step, agent_memory
    
    # Add the memory context to the query if we have any
    if agent_memory.summary or agent_memory.short_term:
        memory_context = agent_memory.get_context()
        enhanced_query = f"{memory_context}\n\nNEW QUERY: {query}\n\nUse memory_tool to recall relevant information if needed."
    else:
        enhanced_query = query
    
    # Run the agent
    result = agent.run(enhanced_query, reset=reset)
    
    # Extract thought and action from the agent's steps
    if hasattr(agent, 'memory') and hasattr(agent.memory, 'steps') and agent.memory.steps:
        for step in agent.memory.steps:
            if hasattr(step, 'thinking') and hasattr(step, 'action'):
                current_step += 1
                thought = step.thinking if step.thinking else "No explicit thought"
                action = str(step.action) if step.action else "No action"
                observation = str(step.observations) if hasattr(step, 'observations') else "No observation"
                
                # Add to memory
                agent_memory.add_interaction(current_step, thought, action, observation)
    
    # Update the summary if we got a response
    if hasattr(result, 'response'):
        # Extract a summary if the response is not just a simple value
        if len(result.response) > 100:  # Only summarize longer responses
            summary_query = "Based on what you just learned, provide a short 1-2 sentence summary of the key information."
            summary_result = agent.run(summary_query)
            if hasattr(summary_result, 'response'):
                agent_memory.update_summary(summary_result.response)
    
    return result

# Create the agent with all tools, replacing wolfram_alpha with math_calculation
agent = CodeAgent(
    model=LiteLLMModel(
        model_id="anthropic/claude-3-sonnet-20240229",
        api_key=api_key,
        temperature=0.5
    ),
    tools=[
        # Web tools
        DuckDuckGoSearchTool(),
        VisitWebpageTool(),
        
        # File tools
        read_pdf_file,
        process_csv,
        
        # Math tools (using internal math library instead of Wolfram Alpha)
        math_calculation,
        
        # Research tools
        get_hugging_face_top_daily_paper,
        
        # Memory tools
        memory_tool
    ],
    system_prompt=enhanced_prompt,
    verbosity_level=2,
    max_steps=20,
    additional_authorized_imports=[
        "os", "requests", "json", "bs4", 
        "pandas", "numpy", "datetime", "math"
    ]
)

# Example GAIA benchmark run function
def run_gaia_question(question):
    print(f"\n\n=== GAIA QUESTION: {question} ===\n")
    response = run_with_memory(agent, question)
    
    # Extract the final answer
    if hasattr(response, 'response'):
        print(f"\nFINAL ANSWER: {response.response}")
        return response.response
    else:
        print(f"\nDIRECT RESULT: {response}")
        return str(response)

# Test with a sample GAIA question
if __name__ == "__main__":
    sample_question = "What is the capital of France?"
    answer = run_gaia_question(sample_question)
    print(f"Answer: {answer}")
    
    # Test memory with a follow-up question
    follow_up = "What is the population of this city?"
    follow_up_answer = run_gaia_question(follow_up)
    print(f"Follow-up answer: {follow_up_answer}")