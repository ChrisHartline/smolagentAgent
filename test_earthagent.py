# test_earthagent.py
from gaia_agent import run_gaia_question
import earthagent  # Import your main agent implementation

# List of GAIA questions to test
gaia_questions = [
    "What is the capital of France?",
    "Calculate the derivative of x^3 * sin(x).",
    "If I have 5 apples and give 2 to my friend, how many do I have left?",
    # Add more GAIA benchmark questions
]

def test_earthagent():
    print("=== TESTING EARTHAGENT WITH GAIA QUESTIONS ===\n")
    
    # Run the questions and collect answers
    answers = {}
    for q in gaia_questions:
        print(f"\nTesting question: {q}")
        answers[q] = run_gaia_question(q)
    
    # Print results
    print("\n=== TEST RESULTS ===\n")
    for q, a in answers.items():
        print(f"QUESTION: {q}")
        print(f"ANSWER: {a}")
        print("-------------------")
    
    # Save results to a file
    with open("gaia_test_results.txt", "w") as f:
        f.write("=== GAIA BENCHMARK TEST RESULTS ===\n\n")
        for q, a in answers.items():
            f.write(f"QUESTION: {q}\n")
            f.write(f"ANSWER: {a}\n\n")
    
    print(f"\nResults saved to gaia_test_results.txt")

if __name__ == "__main__":
    test_earthagent()