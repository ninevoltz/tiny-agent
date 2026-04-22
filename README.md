# tiny-agent

A shell agent for use with Ollama, running local open-weights models. No guardrails or safety checks. Beware. 

Written entirely by Qwen3.5:27b running on Ollama, on my i9-13900k 128GB + RTX 4090 rig.

Notes discovered along the way:

sudo systemctl edit ollama.service

add:

    [Service]
    Environment="OLLAMA_HOST=0.0.0.0:11434"
    Environment="OLLAMA_CONTEXT_LENGTH=1010000"

sudo mkdir /etc/searxng
    
cat > /etc/searxng/settings.yml <<'YAML'
  use_default_settings: true

  search:
    formats:
      - html
      - json
  YAML

sudo chmod -R 666 /etc/searxng


curl http://localhost:11434/api/chat -d '{
  "model": "qwen3.5:27b",
  "messages": [{"role":"user","content":"test"}],
  "stream": false
}'
{"model":"qwen3.5:27b","created_at":"2026-04-22T17:34:19.909316082Z","message":{"role":"assistant","content":"Hello! It seems like you might be testing the system. 😊 How can I assist you today? Whether you have questions, need help with a task, or just want to chat, feel free to let me know!","thinking":"Okay, the user just sent \"test\". Let me figure out why they did that. Maybe they're checking if the system is working, or perhaps they want to see how I respond. Since it's a simple message, I should respond in a friendly and helpful way. Let me keep it light and open-ended so they know I'm here to assist. Maybe offer some examples of what they can ask. Alright, let's put that together."},"done":true,"done_reason":"stop","total_duration":108711285086,"load_duration":119440384,"prompt_eval_count":11,"prompt_eval_duration":940112060,"eval_count":139,"eval_duration":50312270698}
