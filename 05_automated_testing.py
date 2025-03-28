# 📓 05_automated_testing.ipynb — Prompt Evaluation at Scale

import os
import pandas as pd
import time
from openai import OpenAI
from dotenv import load_dotenv

# Load API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -----------------------------------------
# Setup: Models, Prompts, Scoring Criteria
# -----------------------------------------

prompts = [
    "What are the benefits of performance fabric for furniture?",
    "Why is velvet often used in mid-century modern furniture?",
    "What should customers look for when buying eco-friendly upholstery?"
]

models = ["gpt-3.5-turbo", "gpt-4"]
criteria = ["Clarity", "Specificity", "Relevance"]

# -----------------------------------------
# Generate a response from the model
# -----------------------------------------

def generate_response(prompt, model):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=300
    )
    return response.choices[0].message.content

# -----------------------------------------
# Let GPT score each response
# -----------------------------------------

def score_response(prompt, response, criteria):
    scoring_prompt = f"""
Evaluate the following response to the prompt below. Score it from 1–10 on each of these criteria: {', '.join(criteria)}.

Prompt:
{prompt}

Response:
{response}

Return the result as JSON like:
{{
  "Clarity": 8,
  "Specificity": 7,
  "Relevance": 9,
  "Comments": "Brief justification."
}}
"""
    eval_response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an evaluator that scores assistant responses."},
            {"role": "user", "content": scoring_prompt}
        ],
        temperature=0,
        max_tokens=300
    )
    try:
        return eval_response.choices[0].message.content
    except:
        return None

# -----------------------------------------
# Run the full test suite
# -----------------------------------------

results = []

for prompt in prompts:
    for model in models:
        print(f"⏳ Running {model} on: {prompt}")
        response = generate_response(prompt, model)
        time.sleep(1)  # Avoid rate limits
        evaluation = score_response(prompt, response, criteria)
        
        # Parse evaluation JSON (lazy approach for now)
        try:
            score_data = eval(evaluation)  # You could use json.loads() with cleanup
            score_data["Prompt"] = prompt
            score_data["Model"] = model
            score_data["Raw_Response"] = response
            results.append(score_data)
        except Exception as e:
            print("⚠️ Eval parse failed:", e)
            continue

# -----------------------------------------
# Save and Display Results
# -----------------------------------------

df = pd.DataFrame(results)
print("✅ DONE! Here's a preview:")
print(df.head())

# Optional: export
df.to_csv("automated_eval_results.csv", index=False)
