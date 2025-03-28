# 📓 04_eval_testing.ipynb — Prompt Evaluation and Testing

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --------------------------------------------------------
# 1. Compare Prompt Variants (A/B Testing)
# --------------------------------------------------------

def run_prompt(prompt, system_msg="You are a helpful assistant.", model="gpt-4", temperature=0.3):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=300
    )
    return response.choices[0].message.content

# Two prompt styles to compare
prompt_a = "What are the benefits of performance fabric for furniture?"
prompt_b = "List 3 specific reasons why performance fabric is ideal for upholstery."

print("🅰️ Prompt A Result:\n", run_prompt(prompt_a))
print("\n🅱️ Prompt B Result:\n", run_prompt(prompt_b))


# --------------------------------------------------------
# 2. Self-Evaluation Prompt (Ask model to critique)
# --------------------------------------------------------

def self_critique(original_prompt, response_a, response_b):
    evaluation_prompt = f"""
You are an evaluation assistant. Two responses were given to the same prompt. Evaluate both and pick the better one.

Prompt:
{original_prompt}

Response A:
{response_a}

Response B:
{response_b}

Which response is more helpful, clear, and relevant? Justify your answer.
"""
    return run_prompt(evaluation_prompt, system_msg="You are a critical evaluator of AI responses.")

# Example usage
response_a = run_prompt(prompt_a)
response_b = run_prompt(prompt_b)

print("\n🧠 Self-Evaluation:\n")
print(self_critique(prompt_a, response_a, response_b))


# --------------------------------------------------------
# 3. Scoring Heuristics (Custom Evaluation)
# --------------------------------------------------------

def run_scored_eval(prompt, scoring_criteria):
    eval_prompt = f"""
Evaluate the following response based on these criteria: {', '.join(scoring_criteria)}.
Score each out of 10 and give a brief explanation for each.

Prompt:
{prompt}

Response:
{run_prompt(prompt)}
"""
    return run_prompt(eval_prompt, system_msg="You are a scoring assistant.")

criteria = ["Clarity", "Specificity", "Relevance"]
print("\n📊 Scored Evaluation:\n")
print(run_scored_eval(prompt_b, criteria))


# --------------------------------------------------------
# 4. Ground Truth Comparison
# --------------------------------------------------------

def compare_to_ground_truth(prompt, expected_answer):
    model_response = run_prompt(prompt)
    prompt_eval = f"""
Compare the following model response to the ground truth answer.
Point out what's accurate, what's missing, and what's incorrect.

Prompt:
{prompt}

Model Response:
{model_response}

Ground Truth:
{expected_answer}
"""
    return run_prompt(prompt_eval, system_msg="You are a comparison evaluator.")

ground_truth = "Performance fabric is ideal for furniture because it's stain-resistant, durable, and easy to clean."
print("\n🧾 Ground Truth Comparison:\n")
print(compare_to_ground_truth(prompt_a, ground_truth))
