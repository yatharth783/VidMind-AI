from dotenv import dotenv_values
import os

values = dotenv_values(".env")
env_key = os.getenv("SARVAM_API_KEY")
file_key = values.get("SARVAM_API_KEY")

print("========== API KEY CHECK ==========")
print("Project .env exists:", os.path.exists(".env"))
print("Key in .env:", bool(file_key))
print("Key length in .env:", len(file_key.strip()) if file_key else 0)
print("Key in Windows environment:", bool(env_key))
print("Windows environment key length:", len(env_key.strip()) if env_key else 0)
print("===================================")
