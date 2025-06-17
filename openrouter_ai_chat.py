import requests
import json
import os
import socket
import sys

# Configuration (you can set these in your environment variables)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Check if API key is configured
if not OPENROUTER_API_KEY:
    # If not, prompt the user to set it up
    try:
        from dotenv import load_dotenv

        load_dotenv()  # Load environment variables from .env file
        OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    except ImportError:
        # Install dotenv if not available
        import sys

        print(
            "dotenv library not found. Please install it with 'pip install python-dotenv'"
        )
        sys.exit(1)

# Default model for OpenRouter
DEFAULT_MODEL = "deepseek/deepseek-r1-0528-qwen3-8b:free"


class OpenRouterChat:
    def __init__(self, api_key, model=DEFAULT_MODEL):
        self.api_key = api_key
        self.model = model
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        # self.messages = [{"role": "system", "content": "You are a helpful AI assistant."}]
        self.messages = [{"role": "system", "content": "你是一个懂中文的友善的IT专家"}]

    def chat(self):
        print("Welcome to OpenRouter AI Chat!")
        print("Type your message and press Enter. Type 'exit' to quit.\n")

        while True:
            try:
                user_message = input(f"You ({self.model}): ")

                if user_message.lower() in ["exit", "quit"]:
                    print("Goodbye!")
                    break
                elif user_message.lower().startswith("model "):
                    new_model = user_message[6:].strip()
                    self.model = new_model
                    print(f"Switched model to: {self.model}")
                    continue

                # 将用户消息添加到对话历史
                self.messages.append({"role": "user", "content": user_message})

                # Make the API request with streaming enabled
                payload = {
                    "model": self.model,
                    "messages": self.messages,
                    "stream": True,
                }

                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=self.headers,
                    json=payload,
                    stream=True,
                )

                # Handle the response
                if response.status_code == 200:
                    print(
                        f"Assistant ({self.model}): ", end="", flush=True
                    )  # Print the prompt and flush to ensure visibility

                    full_assistant_response = ""  # 用于存储完整的助手回复

                    for line in response.iter_lines():
                        if line:
                            # 解码字节串为 UTF-8 字符串
                            decoded_line = line.decode("utf-8")

                            # OpenRouter/OpenAI API 的流式数据以 "data: " 开头
                            if decoded_line.startswith("data: "):
                                json_str = decoded_line[len("data: ") :].strip()

                                # 处理 [DONE] 标记
                                if json_str == "[DONE]":
                                    break  # 流结束

                                try:
                                    chunk = json.loads(json_str)

                                    if "choices" in chunk and len(chunk["choices"]) > 0:
                                        choice = chunk["choices"][0]
                                        # 检查 'delta' 键，它包含新的内容
                                        if (
                                            "delta" in choice
                                            and "content" in choice["delta"]
                                        ):
                                            content = choice["delta"]["content"]
                                            sys.stdout.write(content)
                                            sys.stdout.flush()
                                            full_assistant_response += content
                                        # 如果是结束块，'finish_reason' 可能在此
                                        elif (
                                            "finish_reason" in choice
                                            and choice["finish_reason"] is not None
                                        ):
                                            pass  # 结束，不需打印内容
                                    elif "error" in chunk:
                                        # 处理流中返回的错误
                                        error_data = chunk.get("error", {})
                                        print(
                                            f"\nError from AI: {error_data.get('message', 'Unknown error')}"
                                        )
                                        full_assistant_response = (
                                            ""  # 清空，表示无有效回复
                                        )
                                        break
                                except json.JSONDecodeError as e:
                                    # 忽略无法解析的行（例如空行或非data:开头的行）
                                    # print(f"Warning: Could not decode JSON chunk: {json_str} - {e}", file=sys.stderr)
                                    continue

                    print()  # Add a newline at the end of the assistant's response

                    # 如果有有效的回复，将其添加到对话历史
                    if full_assistant_response:
                        self.messages.append(
                            {"role": "assistant", "content": full_assistant_response}
                        )
                    else:
                        # 如果没有有效回复，移除用户刚才的输入，避免对话历史不一致
                        if (
                            len(self.messages) > 1
                            and self.messages[-1]["role"] == "user"
                        ):
                            self.messages.pop()

                elif response.status_code == 401:
                    print(
                        "\nAuthentication error (HTTP 401). Please check your API key and try again.\n"
                    )
                    break
                elif response.status_code == 400:
                    print(
                        "\nBad Request (HTTP 400). Please check your request and try again.\n"
                    )
                    break
                elif response.status_code == 429:
                    print("\nRate limit exceeded (HTTP 429). Please try again later.\n")
                    break
                elif response.status_code in [500, 502, 503, 504]:
                    print(
                        f"\nServer error (HTTP {response.status_code}). Please try again.\n"
                    )
                    break
                else:
                    print(f"\nResponse error: {response.status_code} (Unexpected)\n")
                    # We break to avoid further processing of this request
                    break
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except socket.gaierror as e:
                print(
                    f"\nDNS error: {e} (Failed to resolve api.openrouter.ai hostname). Please check your network connection and DNS settings.)"
                )
                break
            except requests.exceptions.RequestException as e:
                print(f"Network request failed: {str(e)}")
                break
            except json.JSONDecodeError as e:
                print(f"Failed to decode the response JSON: {str(e)}")
                break


if __name__ == "__main__":
    chat = OpenRouterChat(OPENROUTER_API_KEY)
    chat.chat()
