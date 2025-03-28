# 📓 01_basics.ipynb — Prompt Engineering Basics

import os
from openai import OpenAI
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------------------------------------
# ZERO-SHOT Prompt (text-davinci-003)
# ---------------------------------------------

def zero_shot_completion(prompt, model="text-davinci-003", temperature=0.7, max_tokens=150):
    response = client.completions.create(
        model=model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].text.strip()

print("🚀 ZERO-SHOT (text-davinci-003):\n")
zero_prompt = "Explain what performance fabric is in simple terms."
print(zero_shot_completion(zero_prompt))


# ---------------------------------------------
# FEW-SHOT Prompt (text-davinci-003)
# ---------------------------------------------

few_shot_prompt = """
Q: What is performance fabric?
A: Performance fabric is a type of material designed to withstand wear, resist stains, and often repel moisture. It's commonly used in furniture, sportswear, and outdoor gear.

Q: What are the advantages of performance fabric for furniture?
A:
"""

print("\n🎯 FEW-SHOT (text-davinci-003):\n")
print(zero_shot_completion(few_shot_prompt))


# ---------------------------------------------
# PROMPT TEMPLATES
# ---------------------------------------------

def format_template(template, **kwargs):
    return template.format(**kwargs)

template = "Write a social media caption for a product called {product}, which is designed for {audience}."
formatted_prompt = format_template(template, product="EcoLuxe Sofa", audience="modern, eco-conscious families")

print("\n📋 PROMPT TEMPLATE:\n")
print(zero_shot_completion(formatted_prompt))


# ---------------------------------------------
# RESPONSE FORMATTING (bullets & JSON)
# ---------------------------------------------

# Ask GPT to return a formatted list
formatting_prompt = """
List 3 key selling points of performance fabric furniture. Format the answer as bullet points.
"""

print("\n🧾 BULLET POINT FORMAT:\n")
print(zero_shot_completion(formatting_prompt))

# Ask GPT to return JSON
json_prompt = """
Provide a JSON object with keys: "benefits", "popular_brands", and "price_range" for performance fabric furniture.
"""

print("\n📦 JSON FORMAT:\n")
print(zero_shot_completion(json_prompt))


# ---------------------------------------------
# Optional: Run Similar Tasks with Chat (gpt-3.5-turbo)
# ---------------------------------------------

def chat_prompt(prompt, system_msg="You are a helpful assistant.", model="gpt-3.5-turbo"):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

print("\n💬 SAME TASK WITH gpt-3.5-turbo:\n")
chat_version = "What are the top 3 benefits of using performance fabric in upholstery?"
print(chat_prompt(chat_version))
