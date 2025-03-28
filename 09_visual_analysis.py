#%%
# 📓 09_visual_analysis.ipynb — Prompt Evaluation with Charts, Scoring, and Export
import os
import pandas as pd
import openai
import time
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

# Load API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ---------------------------------------
#%%
# Configuration
# ---------------------------------------

prompts = [
    "Describe the benefits of performance fabric in furniture design.",
    "Why is velvet popular in mid-century modern interiors?",
    "What makes eco-friendly upholstery attractive to modern buyers?"
]

models = ["gpt-3.5-turbo", "gpt-4"]
temperature = 0.7
max_tokens = 200

# Optional: Add manual human ratings here (per prompt, per model)
human_scores = {
    # Format: (prompt, model): {"Clarity": int, "Specificity": int, "Verbosity": int}
    ("Describe the benefits of performance fabric in furniture design.", "gpt-4"): {"Clarity": 9, "Specificity": 8, "Verbosity": 7},
    ("Describe the benefits of performance fabric in furniture design.", "gpt-3.5-turbo"): {"Clarity": 7, "Specificity": 6, "Verbosity": 6},
    # Add more if desired
}

# ---------------------------------------
#%%
# Helpers
# ---------------------------------------

def count_words(text):
    return len(text.split())

def count_sentences(text):
    return len([s for s in text.split('.') if len(s.strip()) > 0])

def get_response(prompt, model):
    chat = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return chat.choices[0].message.content.strip()

def score_with_gpt(prompt, response):
    scoring_prompt = f"""
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
    chat = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an evaluator of assistant responses."},
            {"role": "user", "content": scoring_prompt}
        ],
        temperature=0
    )
    try:
        return eval(chat.choices[0].message.content.strip())
    except:
        return {"Clarity": None, "Specificity": None, "Verbosity": None, "Comments": "Parse failed"}

# ---------------------------------------
#%%
# Run All Evaluations
# ---------------------------------------

results = []

for prompt in prompts:
    for model in models:
        print(f"⏳ Running {model} on: {prompt[:40]}...")
        response = get_response(prompt, model)
        gpt_score = score_with_gpt(prompt, response)
        human_score = human_scores.get((prompt, model), {})

        results.append({
            "Prompt": prompt,
            "Model": model,
            "Response": response,
            "GPT_Clarity": gpt_score.get("Clarity"),
            "GPT_Specificity": gpt_score.get("Specificity"),
            "GPT_Verbosity": gpt_score.get("Verbosity"),
            "GPT_Comments": gpt_score.get("Comments"),
            "Human_Clarity": human_score.get("Clarity"),
            "Human_Specificity": human_score.get("Specificity"),
            "Human_Verbosity": human_score.get("Verbosity"),
            "Word_Count": count_words(response),
            "Sentence_Count": count_sentences(response)
        })

df = pd.DataFrame(results)

# ---------------------------------------
#%%
# Charting
# ---------------------------------------

sns.set(style="whitegrid")

def plot_metric(metric, source):
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="Model", y=f"{source}_{metric}", hue="Prompt")
    plt.title(f"{metric} ({source}) by Model and Prompt")
    plt.ylabel(metric)
    plt.show()

plot_metric("Clarity", "GPT")
plot_metric("Specificity", "GPT")
plot_metric("Verbosity", "GPT")
plot_metric("Clarity", "Human")
plot_metric("Specificity", "Human")

# Word Count Comparison
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="Model", y="Word_Count", hue="Prompt")
plt.title("📝 Word Count by Model and Prompt")
plt.show()

# ---------------------------------------
#%%
# Export
# ---------------------------------------

df.to_csv("09_model_eval_dashboard_data.csv", index=False)
print("📁 Exported to 09_model_eval_dashboard_data.csv")
