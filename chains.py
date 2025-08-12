from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# Prompt for tweet generation
generation_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a techie influencer assistant tasked with writing excellent twitter posts. "
     "Generate the best twitter post possible for the users' request. "
     "If the user provides critique, respond with a revised version of your previous attempts."),
    MessagesPlaceholder(variable_name="messages")  # Accepts the conversation history
])

# Prompt for tweet critique
reflection_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a viral twitter influencer grading a tweet. "
     "Generate critique and recommendations for the user's tweet. "
     "Always provide detailed recommendations, including requests for length, virality, style, etc."),
    MessagesPlaceholder(variable_name="messages")
])

# LLM instance
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Chains
generation_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm

# Optional direct test
test_messages = [{"role": "user", "content": "Write a tweet about the latest tech trends."}]
response = generation_chain.invoke({"messages": test_messages})
print("Initial Response:", response.content)
