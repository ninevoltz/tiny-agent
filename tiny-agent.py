import argparse
import os
import re
import subprocess
import sys
import atexit
import json
import requests
import base64

try:
    import ollama
    HAS_OLLAMA_CLIENT = True
except ImportError:
    HAS_OLLAMA_CLIENT = False

try:
    from PIL import Image
    HAS_PILLOW = True
except ImportError:
    HAS_PILLOW = False

try:
    from prompt_toolkit import prompt
    from prompt_toolkit.history import FileHistory
    HAS_PROMPT_TOOLKIT = True
except ImportError:
    HAS_PROMPT_TOOLKIT = False
    import readline

# Configuration Constants
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen3.6:35b"
DEFAULT_SEARXNG_URL = "http://localhost:8888/"
MAX_AGENT_LOOPS = 100
SHELL_TIMEOUT = 600
SEARCH_TIMEOUT = 30
MAX_SEARCH_RESULTS = 25

FUNCTION_ACTIVE = False

def setup_readline():
    """Configure standard readline if prompt_toolkit is not available."""
    if not HAS_PROMPT_TOOLKIT:
        hist_file = os.path.expanduser("~/.tiny_agent_history")
        readline.read_history_file(hist_file)
        readline.set_history_length(100)
        atexit.register(readline.write_history_file, hist_file)

def get_user_input(prompt_text="> "):
    """Handles user input with history support."""
    if HAS_PROMPT_TOOLKIT:
        hist_file = os.path.expanduser("~/.tiny_agent_history")
        return prompt(prompt_text, history=FileHistory(hist_file))
    else:
        return input(prompt_text)

def create_ollama_client(base_url):
    """Create an Ollama client with custom base URL."""
    return ollama.Client(host=base_url)

def stream_ollama_response(client, model, messages, system_prompt):
    """Sends messages to Ollama and streams the response."""
    chat_messages = [
        {"role": "system", "content": system_prompt}
    ] + messages

    full_content = ""
    try:
        stream = client.chat(
            model=model,
            messages=chat_messages,
            stream=True
        )

        for chunk in stream:
            content = chunk['message']['content']
            full_content += content
            sys.stdout.write(content)
            sys.stdout.flush()
    except Exception as e:
        print(f"\n[Error] API Request failed: {e}")
        return None

    sys.stdout.write("\n")
    return full_content

def encode_image_for_ollama(image_path):
    """Opens an image file and returns it as a base64 string."""
    if not HAS_PILLOW:
        return None
    try:
        if not os.path.isfile(image_path):
            return None
        with open(image_path, 'rb') as f:
            encoded = base64.b64encode(f.read()).decode('utf-8')
        return encoded
    except Exception as e:
        print(f"[Error] Failed to encode image {image_path}: {e}")
        return None

def execute_shell_command(command):
    global FUNCTION_ACTIVE
    """Executes a shell command securely with timeout."""
    print(f"\n[Agent] Executing: {command}\n")
    try:
        process = subprocess.Popen(
            f"bash -c '{command}'",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        output, _ = process.communicate(timeout=SHELL_TIMEOUT)
        FUNCTION_ACTIVE = False
        return True, output.strip()
    except subprocess.TimeoutExpired:
        FUNCTION_ACTIVE = False
        return False, f"Command timed out (exceeded {SHELL_TIMEOUT}s)."
    except Exception as e:
        FUNCTION_ACTIVE = False
        return False, f"Command execution failed: {str(e)}"

def parse_shell_command(content):
    global FUNCTION_ACTIVE
    """Extracts the shell command if present."""
    pattern = r'<do_shell_command>(.*?)</do_shell_command>'
    match = re.search(pattern, content, re.DOTALL)
    if match and not FUNCTION_ACTIVE:
        FUNCTION_ACTIVE = True
        return match.group(1).strip()
    return None

def execute_web_search(query, searxng_url, max_results=MAX_SEARCH_RESULTS):
    """
    Executes a web search using SearXNG API.
    """
    global FUNCTION_ACTIVE
    print(f"\n[Agent] Searching: {query}\n")
    search_url = f"{searxng_url}/search?q={query}&format=json"
    try:
        response = requests.get(
            search_url,
            timeout=SEARCH_TIMEOUT,
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        if response.status_code != 200:
            return False, f"Search request failed with status {response.status_code}"

        data = response.json()
        results = data.get('results', [])[:max_results]

        if not results:
            return False, "No search results found."

        formatted_results = []
        for i, result in enumerate(results, 1):
            title = result.get('title', 'No title')
            url = result.get('url', 'No URL')
            snippet = result.get('content', 'No description')[:200]
            formatted_results.append(f"[{i}] {title}\nURL: {url}\nSnippet: {snippet}\n")

        output = "Search Results:\n" + "=" * 50 + "\n" + "\n".join(formatted_results)
        print(f"{output}\n" + "=" * 50 + "\n")
        FUNCTION_ACTIVE = False
        return True, output

    except requests.exceptions.Timeout:
        FUNCTION_ACTIVE = False
        return False, f"Search request timed out (exceeded {SEARCH_TIMEOUT}s)."
    except json.JSONDecodeError:
        FUNCTION_ACTIVE = False
        return False, "SearXNG did not return valid JSON."
    except Exception as e:
        FUNCTION_ACTIVE = False
        return False, f"Search execution failed: {str(e)}"

def parse_web_search(content):
    """Extracts the web search query if present."""
    global FUNCTION_ACTIVE
    pattern = r'<do_web_search>(.*?)</do_web_search>'
    match = re.search(pattern, content, re.DOTALL)
    if match and not FUNCTION_ACTIVE:
        FUNCTION_ACTIVE = True
        return match.group(1).strip()
    return None

def parse_view_image(content):
    """Extracts the image path if present."""
    global FUNCTION_ACTIVE
    pattern = r'<do_view_image>(.*?)</do_view_image>'
    match = re.search(pattern, content, re.DOTALL)
    if match and not FUNCTION_ACTIVE:
        FUNCTION_ACTIVE = True
        return match.group(1).strip()
    return None

def main():
    parser = argparse.ArgumentParser(description="Ollama CLI Harness with Vision, Shell Execution and Web Search")
    parser.add_argument('--base-url', default=DEFAULT_OLLAMA_URL,
                       help=f"Ollama API URL (default: {DEFAULT_OLLAMA_URL})")
    parser.add_argument('--model', default=DEFAULT_MODEL,
                       help=f"Model name (default: {DEFAULT_MODEL})")
    parser.add_argument('--searxng-url', default=DEFAULT_SEARXNG_URL,
                       help=f"SearXNG API URL (default: {DEFAULT_SEARXNG_URL})")
    args = parser.parse_args()

    ollama_base_url = args.base_url
    model = args.model
    searxng_url = args.searxng_url

    if not HAS_OLLAMA_CLIENT:
        print("[Error] The ollama Python client is not installed.")
        print("Please install it with: pip install ollama")
        sys.exit(1)

    if not HAS_PILLOW:
        print("[Warning] Pillow is not installed. Image viewing will be disabled.")
        print("Install it with: pip install pillow")

    setup_readline()

    # System Prompt updated for Vision
    system_instruction = (
        "You have access to the user's command line shell, the internet, and the ability to view images.\n"
        "1. For shell tasks, wrap commands in <do_shell_command>COMMAND_HERE</do_shell_command>\n"
        "2. For web search, wrap queries in <do_web_search>QUERY_HERE</do_web_search>\n"
        "3. To view an image file, wrap the file path in <do_view_image>PATH_HERE</do_view_image>\n"
        "I will execute these automatically and provide the output to you.\n"
        "After execution, I will send <agent_may_continue>.\n"
        "Do not wait for user input unless you need more information or have finished the task.\n"
        "Ensure shell commands are safe. Do not run destructive commands like 'rm -rf'."
    )

    messages = []
    client = create_ollama_client(ollama_base_url)

    print(f"\n[Info] Connected to Ollama at {ollama_base_url} using model {model}")
    print(f"[Info] Connected to SearXNG at {searxng_url}")
    print(f"[Info] Type 'exit' to quit.")
    print(f"[Info] Agent will execute <do_shell_command>, <do_web_search>, and <do_view_image>...\n")

    print(f"[Info] Note: Ensure SearXNG has JSON format enabled in settings.yml")
    print(f"[Info] If you get 403 errors, check SearXNG documentation\n")

    try:
        while True:
            human_input = get_user_input("You > ")
            if human_input.lower() == 'exit':
                print("Goodbye!")
                break

            if not human_input.strip():
                continue

            # User input acts as a normal text message
            messages.append({"role": "user", "content": human_input})

            agent_loop_count = 0

            while True:
                response_content = stream_ollama_response(client, model, messages, system_instruction)

                if not response_content:
                    break

                messages.append({"role": "assistant", "content": response_content})

                # Check for Shell Command
                shell_cmd = parse_shell_command(response_content)

                # Check for Web Search
                web_search_query = parse_web_search(response_content)

                # Check for View Image
                image_path = parse_view_image(response_content)

                if shell_cmd or web_search_query or image_path:
                    agent_loop_count += 1
                    if agent_loop_count > MAX_AGENT_LOOPS:
                        print(f"\n[Warning] Max agent loops ({MAX_AGENT_LOOPS}) reached. Returning control to user.")
                        break

                    if shell_cmd:
                        success, output = execute_shell_command(shell_cmd)
                        if output:
                            agent_feedback = f"{output}\n<agent_may_continue>"
                            messages.append({
                                "role": "user",
                                "content": agent_feedback
                            })
                            continue

                    if web_search_query:
                        success, output = execute_web_search(web_search_query, searxng_url)
                        if output:
                            agent_feedback = f"{output}\n<agent_may_continue>"
                            messages.append({
                                "role": "user",
                                "content": agent_feedback
                            })
                            continue

                    if image_path:
                        if HAS_PILLOW:
                            b64_image = encode_image_for_ollama(image_path)
                            if b64_image:
                                # Send the image to the model
                                messages.append({
                                    "role": "user",
                                    "content": f"Viewed image from path: {image_path}",
                                    "images": [b64_image]
                                })
                                # No text feedback needed, just continue loop to ask model what it sees
                                continue
                            else:
                                print(f"[Error] Failed to view image {image_path}")
                        else:
                            print("[Error] Image viewing is not available (Pillow missing).")

                        # Even if image failed, we should break to avoid infinite loop if we can't process it
                        break

                # No command detected, finish turn
                break

    except KeyboardInterrupt:
        print("\n[Info] Interrupted. Goodbye.")
        sys.exit(0)

if __name__ == "__main__":
    main()
