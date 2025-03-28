# 02_chat_format.ipynb

# 🧠 Prompt Engineering with Chat Format (GPT-3.5/4)
# Experimenting with role prompting, temperature, prompt chaining, and output parsing

import os
from openai import OpenAI
from dotenv import load_dotenv
import json

# Load API key from .env file
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ----------------------------------------------------
# Step 1: Basic Chat Format with Roles
# ----------------------------------------------------

def basic_chat(prompt, model="gpt-4", temperature=0.7, max_tokens=300):
    messages = [
        {"role": "system", "content": "You are a helpful assistant that gives concise, structured answers."},
        {"role": "user", "content": prompt}
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content

prompt = "Summarize the company Patagonia in 3 bullet points."
print("📌 Basic Chat Response:\n", basic_chat(prompt))


# ----------------------------------------------------
# Step 2: Iterative Refinement (Chain of Prompts)
# ----------------------------------------------------

# Step 2A: Ask for a basic summary
summary_prompt = "Give a 1-sentence summary of the company Crypton Fabric."
summary = basic_chat(summary_prompt)

# Step 2B: Use the output to refine further
refinement_prompt = f"Based on this summary: '{summary}', extract 3 unique competitive advantages."
refined = basic_chat(refinement_prompt)

print("\n🔁 Iterative Refinement:\n", refined)


# ----------------------------------------------------
# Step 3: Experiment with Temperature
# ----------------------------------------------------

print("\n🔥 Temperature Comparison:\n")

for temp in [0.2, 0.7, 1.0]:
    result = basic_chat("Give a creative product name for a new line of sustainable outdoor fabrics.", temperature=temp)
    print(f"Temperature {temp} → {result}")


# ----------------------------------------------------
# Step 4: Parse Output into Structured JSON
# ----------------------------------------------------

def get_company_profile(company_name):
    messages = [
        {"role": "system", "content": "You are a market analyst. Respond in valid JSON format with keys: name, hq_location, founded, specialties."},
        {"role": "user", "content": f"Provide a company profile for {company_name}."}
    ]
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0.3,
        max_tokens=300
    )
    raw_output = response.choices[0].message.content
    try:
        data = json.loads(raw_output)
    except json.JSONDecodeError:
        print("⚠️ JSON parsing failed. Raw output:\n", raw_output)
        return None
    return data

profile = get_company_profile("Keyston Brothers")
print("\n📦 Parsed Company Profile:\n", profile)


# ----------------------------------------------------
# Step 5: Prompt Chaining for Data Transformation
# ----------------------------------------------------

# Example: take structured data and generate a pitch
def generate_pitch(profile):
    prompt = f"""Using this profile, generate a 2-sentence sales pitch:
{json.dumps(profile, indent=2)}
"""
    return basic_chat(prompt)

if profile:
    pitch = generate_pitch(profile)
    print("\n🧩 Prompt Chaining → Sales Pitch:\n", pitch)
