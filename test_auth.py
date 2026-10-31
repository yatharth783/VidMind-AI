from dotenv import load_dotenv
import os
from sarvamai import SarvamAI

load_dotenv(override=True)

api_key = os.getenv("SARVAM_API_KEY")

print("API key loaded:", bool(api_key))
print("API key length:", len(api_key) if api_key else 0)

client = SarvamAI(api_subscription_key=api_key)

print("Sarvam client initialized successfully!")
