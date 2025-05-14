# combined_test.py
import os
import sys
from gaia_agent import run_gaia_question, agent_memory

# GAIA benchmark questions
gaia_questions = [
    "What is the capital of France?",
    "What is the square root of 144?",
    "What is the current population of Tokyo?",
    "What are the first 5 prime numbers?",
    "Who wrote the novel 'Pride and Prejudice'?"
]

def run_benchmark():
    print("=== RUNNING GAIA BENCHMARK TESTS ===\n")
    
    results = {}
    for i, question in enumerate(gaia_questions):
        print(f"Running question {i+1}/{len(gaia_questions)}: {question}")
        answer = run_gaia_question(question)
        results[question] = answer
    
    # Display results
    print("\n=== RESULTS ===\n")
    for q, a in results.items():
        print(f"Q: {q}")
        print(f"A: {a}")
        print("-" * 50)
    
    # Save results to file
    with open("benchmark_results.txt", "w") as f:
        f.write("=== GAIA BENCHMARK RESULTS ===\n\n")
        for q, a in results.items():
            f.write(f"Q: {q}\n")
            f.write(f"A: {a}\n\n")
    
    print(f"\nResults saved to benchmark_results.txt")

if __name__ == "__main__":
    run_benchmark()