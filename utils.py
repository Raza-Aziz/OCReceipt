from dotenv import load_dotenv
from groq import Groq
import os

load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Initialize Groq client
def init_groq():
    api_key = GROQ_API_KEY
    if api_key:
        return Groq(api_key=api_key)
    return None