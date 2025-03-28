# 📓 03_advanced_strategies.ipynb — Advanced Prompt Engineering

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ----------------------------------------------------
# 1. Custom Persona (System Role)
# ----------------------------------------------------

def run_persona_query(prompt, persona, model="gpt-4", temperature=0.3):
    messages = [
        {"role": "system", "content": persona},
        {"role": "user", "content": prompt}
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=300
    )
    return response.choices[0].message.content

persona = "You are a textile historian with expertise in vintage American upholstery trends."
prompt = "Explain why velvet was popular in mid-century modern furniture."

print("🧠 Custom Persona:\n", run_persona_query(prompt, persona))


# ----------------------------------------------------
# 2. Multi-Step Workflow (Chained Tasks)
# ----------------------------------------------------

def generate_keywords(product):
    step1_prompt = f"Generate 5 SEO keywords for a product called '{product}' in the interior design space."
    return run_persona_query(step1_prompt, "You are an SEO marketing assistant.")

def generate_caption(keywords):
    step2_prompt = f"Write an Instagram caption using the keywords: {', '.join(keywords)}."
    return run_persona_query(step2_prompt, "You are a witty brand copywriter.")

print("\n🔁 Multi-Step Workflow:")
keywords = generate_keywords("LuxeTweed Performance Chair").split("\n")
print("Step 1 - Keywords:", keywords)
caption = generate_caption(keywords)
print("Step 2 - Caption:\n", caption)


# ----------------------------------------------------
# 3. Structured Output (Schema-based JSON)
# ----------------------------------------------------

def get_structured_analysis(company_name):
    system_msg = "You are a market analyst. Return your response in JSON with fields: company, location_count, primary_markets, notes."
    user_msg = f"Analyze the company '{company_name}' for a competitor research report."

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        temperature=0.3,
        max_tokens=300
    )

    raw = response.choices[0].message.content
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        print("⚠️ Invalid JSON. Raw output:\n", raw)
        return None

print("\n📦 Structured Output (JSON):")
print(get_structured_analysis("Keyston Brothers"))


# ----------------------------------------------------
# 4. Function Calling (Tool Use)
# ----------------------------------------------------
# Available only with gpt-4-0613 or gpt-3.5-turbo-0613+

def run_function_call():
    def define_tool_schema():
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_fabric_care",
                    "description": "Provides care instructions for a fabric type.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "fabric_type": {
                                "type": "string",
                                "description": "Type of fabric (e.g., velvet, tweed, linen)"
                            }
                        },
                        "required": ["fabric_type"]
                    }
                }
            }
        ]

    tools = define_tool_schema()

    response = client.chat.completions.create(
        model="gpt-4-0613",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "How do I care for linen upholstery?"}
        ],
        tools=tools,
        tool_choice="auto"
    )

    print("\n🔧 Function Calling Response:\n")
    print(response.choices[0].message)

# Uncomment if using a model that supports function calling
# run_function_call()
