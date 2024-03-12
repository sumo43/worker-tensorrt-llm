import openai

# Set up your API key and base URL
openai.api_key = "YOUR_API_KEY"
openai.api_base = "http://localhost:8887/v1"

# Set up the request data
data = {
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Hello! How are you today?"}],
    "stream": True
}

# Send the request to the API endpoint
response = openai.ChatCompletion.create(**data)

# Process the streaming response
for chunk in response:
    chunk_message = chunk['choices'][0]['delta']
    if 'content' in chunk_message:
        content = chunk_message['content']
        print(content, end='', flush=True)

print("\nStreaming completed.")
