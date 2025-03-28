# 📓 06_langchain_intro.ipynb — LangChain Basics

# --------------------------------------------------------
#%%
# 1. Setup
# --------------------------------------------------------

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4", temperature=0.3)

# --------------------------------------------------------
#%%
# 2. Basic LLMChain (PromptTemplate → LLM → Output)
# --------------------------------------------------------

template = PromptTemplate.from_template("What are the benefits of {product} in home design?")
chain = template | llm | StrOutputParser()

response = chain.invoke({"product": "performance fabric"})
print("🧠 Basic LLMChain Response:\n", response)

# --------------------------------------------------------
#%%
# 3. Sequential Chaining
# --------------------------------------------------------

# First step: get keywords
keywords_prompt = PromptTemplate.from_template("Generate 5 SEO keywords for {product}")
keywords_chain = keywords_prompt | llm | StrOutputParser()

# Second step: write caption
caption_prompt = PromptTemplate.from_template("Write an Instagram caption using these keywords: {keywords}")
caption_chain = caption_prompt | llm | StrOutputParser()

# Full flow:
product = "EcoLuxe Linen Sofa"
keywords = keywords_chain.invoke({"product": product})
caption = caption_chain.invoke({"keywords": keywords})

print("\n🔁 Multi-Step Chain Result:\n", caption)

# --------------------------------------------------------
#%%
# 4. Output Parsing (JSON style)
# --------------------------------------------------------

from langchain_core.output_parsers import JsonOutputParser

json_prompt = PromptTemplate.from_template(
    "Return a JSON object with keys: 'brand', 'eco_features', and 'price_range' for {product}"
)

json_chain = json_prompt | llm | JsonOutputParser()

print("\n📦 Structured Output:\n", json_chain.invoke({"product": "Sunbrella Performance Fabric"}))


# --------------------------------------------------------
#%%
# 5. Document Loading (TextLoader)
# --------------------------------------------------------

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

# Load and split file
loader = TextLoader("sample_data/fabric_article.txt")  # Add your own .txt file
docs = loader.load()

splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=30)
chunks = splitter.split_documents(docs)

print("\n📚 Loaded and Split Docs:\n")
print(chunks[0].page_content[:300])  # Preview

# Optional: Use this for retrieval in RAG later!
