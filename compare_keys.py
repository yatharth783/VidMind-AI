from dotenv import dotenv_values
import os

file_key = dotenv_values(".env").get("SARVAM_API_KEY", "")
env_key = os.getenv("SARVAM_API_KEY", "")

print("Both keys match:", file_key.strip() == env_key.strip())
print("File key starts with whitespace:", file_key != file_key.strip())
print("Environment key starts with whitespace:", env_key != env_key.strip())
