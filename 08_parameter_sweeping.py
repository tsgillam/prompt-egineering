# 📓 08_parameter_sweep.ipynb — Supercharged Parameter Sweep

import os
import pandas as pd
import time
import openai
from dotenv import load_dotenv

# Load API key
load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------------------------------
#%%
# Config: Prompts & Sweep Parameters
# ---------------------------------------

prompts = [
    "Describe the benefits of performance fabric in furniture design.",
    "Why is velvet popular in mid-century modern interiors?",
    "What makes eco-friendly upholstery attractive to modern buyers?"
]

temperature_values = [0.2, 0.7, 1.0]
max_token_values = [50, 150, 300]
model = "gpt-4"

# ---------------------------------------
#%%
# Helpers: Token + Word Count
# ---------------------------------------

import re

def count_words(text):
    return len(re.findall(r"\w+", text))

def count_sentences(text):
    return len(re.findall(r'[.!?]', text))

# ---------------------------------------
#%%
# Run GPT and Evaluate Itself
# ---------------------------------------

def generate_response(prompt, temperature, max_tokens):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content.strip()

def self_score(prompt, response):
    eval_prompt = f"""
Evaluate the response to this prompt:
Prompt: "{prompt}"

Response:
{response}

Score the response from 1–10 in the following categories:
- Clarity
- Specificity
- Verbosity

Return JSON like:
{{
  "Clarity": 8,
  "Specificity": 7,
  "Verbosity": 6,
  "Comments": "Short and precise, but lacks vivid examples."
}}
"""
    eval_response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a strict evaluator of model outputs."},
            {"role": "user", "content": eval_prompt}
        ],
        temperature=0,
        max_tokens=300
    )
    try:
        return eval(eval_response.choices[0].message.content)
    except:
        return {"Clarity": None, "Specificity": None, "Verbosity": None, "Comments": "Failed to parse"}

# ---------------------------------------
#%%
# Main Sweep Loop
# ---------------------------------------

results = []

for prompt in prompts:
    for temp in temperature_values:
        for max_tokens in max_token_values:
            print(f"⚙️ Running temp={temp}, tokens={max_tokens} on prompt: {prompt[:40]}...")
            response = generate_response(prompt, temp, max_tokens)
            time.sleep(1)  # Rate-limit kindness

            word_count = count_words(response)
            sentence_count = count_sentences(response)
            score = self_score(prompt, response)

            results.append({
                "Prompt": prompt,
                "Temperature": temp,
                "Max Tokens": max_tokens,
                "Response": response,
                "Clarity": score.get("Clarity"),
                "Specificity": score.get("Specificity"),
                "Verbosity": score.get("Verbosity"),
                "Comments": score.get("Comments"),
                "Word Count": word_count,
                "Sentence Count": sentence_count
            })

# ---------------------------------------
#%%
# Display and Export
# ---------------------------------------

df = pd.DataFrame(results)
pd.set_option('display.max_colwidth', None)

print("✅ Done! Sample output:")
# display(df.head())

# Save for deeper analysis
df.to_csv("sweep_eval_results.csv", index=False)
