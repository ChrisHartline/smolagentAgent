from earthagent import agent

def run_gaia_question(question):
    """
    Process a GAIA benchmark question using the agent.
    
    Args:
        question: The GAIA question to process
    
    Returns:
        The agent's answer to the question
    """
    try:
        result = agent.run(question)
        return result.response if hasattr(result, 'response') else str(result)
    except Exception as e:
        return f"Error processing question: {str(e)}"

# ============================================ GAIA Questions ============================================ #
gaia_questions = [
    "What is the capital of France?",
    "Calculate the derivative of x^3 * sin(x).",
    "If I have 5 apples and give 2 to my friend, how many do I have left?",
    # Add more GAIA benchmark questions
]

# Run the questions and collect answers
answers = {}
for q in gaia_questions:
    answers[q] = run_gaia_question(q)

# Save results to a file
with open("gaia_results.txt", "w") as f:
    f.write("=== GAIA BENCHMARK RESULTS ===\n\n")
    for q, a in answers.items():
        f.write(f"QUESTION: {q}\n")
        f.write(f"ANSWER: {a}\n\n")