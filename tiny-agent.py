import argparse
import os
import re
import subprocess
import sys
import atexit

try:
    import ollama
    HAS_OLLAMA_CLIENT = True
except ImportError:
    HAS_OLLAMA_CLIENT = False

try:
    from prompt_toolkit import prompt
    from prompt_toolkit.history import FileHistory
    HAS_PROMPT_TOOLKIT = True
except ImportError:
    HAS_PROMPT_TOOLKIT = False
    import readline

# Configuration Constants
DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen3.5:27b"  # Ensure this model is pulled in Ollama
MAX_AGENT_LOOPS = 5       # Prevent infinite command loops
SHELL_TIMEOUT = 10        # Seconds allowed for a shell command

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
    # The Ollama client supports custom base URL configuration [1]
    return ollama.Client(host=base_url)

def stream_ollama_response(client, model, messages, system_prompt):
    """Sends messages to Ollama and streams the response using the official client."""
    # Configure messages with system prompt
    chat_messages = [
        {"role": "system", "content": system_prompt}
    ] + messages

    full_content = ""
    try:
        # Use the streaming feature from the ollama Python library [11]
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

def execute_shell_command(command):
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
        return True, output.strip()
    except subprocess.TimeoutExpired:
        return False, f"Command timed out (exceeded {SHELL_TIMEOUT}s)."
    except Exception as e:
        return False, f"Command execution failed: {str(e)}"

def parse_shell_command(content):
    """Extracts the shell command if present."""
    pattern = r'<do_shell_command>(.*?)</do_shell_command>'
    match = re.search(pattern, content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def main():
    parser = argparse.ArgumentParser(description="Ollama CLI Harness with Shell Execution")
    parser.add_argument('--base-url', default=DEFAULT_BASE_URL,
                       help=f"Ollama API URL (default: {DEFAULT_BASE_URL})")
    parser.add_argument('--model', default=DEFAULT_MODEL,
                       help=f"Model name (default: {DEFAULT_MODEL})")
    args = parser.parse_args()

    base_url = args.base_url
    model = args.model

    # Verify the ollama client is available
    if not HAS_OLLAMA_CLIENT:
        print("[Error] The ollama Python client is not installed.")
        print("Please install it with: pip install ollama")
        sys.exit(1)

    # Setup CLI History
    setup_readline()

    # System Prompt Definition
    system_instruction = (
        "You are an AI assistant with access to the user's command line shell.\n"
        "If you need to perform a shell task, wrap the command in <do_shell_command>COMMAND_HERE</do_shell_command>.\n"
        "I will execute this command automatically.\n"
        "The output of the command will be provided to you immediately.\n"
        "After the shell output is provided, I will send <agent_may_continue>.\n"
        "Do not wait for user input unless you need more information or have finished the task.\n"
        "Ensure commands are safe. Do not run destructive commands like 'rm -rf'."
    )

    # Initialize Conversation History
    messages = []

    # Create the Ollama client with custom base URL [4]
    client = create_ollama_client(base_url)

    print(f"\n[Info] Connected to Ollama at {base_url} using model {model}\n"
          f"[Info] Type 'exit' to quit.\n"
          f"[Info] Agent will execute commands wrapped in <do_shell_command>...\n")

    try:
        while True:
            # Get User Input
            human_input = get_user_input("You > ")
            if human_input.lower() == 'exit':
                print("Goodbye!")
                break

            if not human_input.strip():
                continue

            # Add user message to history
            messages.append({"role": "user", "content": human_input})

            # Agent Loop - allows multiple shell commands in a row
            agent_loop_count = 0

            while True:
                # Call Ollama using the official client library [9]
                response_content = stream_ollama_response(client, model, messages, system_instruction)

                if not response_content:
                    break

                # Add assistant response to history
                messages.append({"role": "assistant", "content": response_content})

                # Check for Shell Command
                shell_cmd = parse_shell_command(response_content)

                if shell_cmd:
                    agent_loop_count += 1
                    if agent_loop_count > MAX_AGENT_LOOPS:
                        print(f"\n[Warning] Max agent loops ({MAX_AGENT_LOOPS}) reached. Returning control to user.")
                        break

                    print("-" * 40)
                    # Execute the command
                    success, output = execute_shell_command(shell_cmd)
                    print(f"Result: {output}\n" + "-" * 40)

                    # Prepare the response to send back to the model
                    agent_feedback = f"{output}\n<agent_may_continue>"

                    # Add feedback to history
                    messages.append({
                        "role": "user",
                        "content": agent_feedback
                    })

                    # Continue loop for next instruction
                    continue
                else:
                    # No command detected, finish turn
                    break

    except KeyboardInterrupt:
        print("\n[Info] Interrupted. Goodbye.")
        sys.exit(0)

if __name__ == "__main__":
    main()
