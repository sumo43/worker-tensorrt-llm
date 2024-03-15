curl -X POST https://api.runpod.ai/v2/your_endpoint_id/run               \
-H 'Content-Type: application/json'                                 \
-H 'Authorization: Bearer ' \
-d '{"input": {"number": 2}}'
openai.api_base = "http://localhost:8887/v1"

