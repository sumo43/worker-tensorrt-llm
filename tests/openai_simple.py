import requests
import json

# Set up the API endpoint and your API key
api_endpoint = "http://localhost:8887/v1/chat/completions"
api_key = "YOUR_API_KEY"

# Set up the request headers
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

# Set up the request data
data = {
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Hello! How are you today?"}],
    "stream": True
}

# Send the request to the API endpoint
response = requests.post(api_endpoint, headers=headers, data=json.dumps(data), stream=True)

# Process the streaming response
for line in response.iter_lines():
    if line:
        decoded_line = line.decode("utf-8")
        if "data: " in decoded_line:
            message = decoded_line[6:]
            if message != "[DONE]":
                chat_message = json.loads(message)
                content = chat_message["choices"][0]["delta"].get("content", "")
                print(content, end="", flush=True)

print("\nStreaming completed.")
