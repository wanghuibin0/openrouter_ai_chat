#!/usr/bin/env python

import requests
import json
import os
import socket
import sys

# Configuration (you can set these in your environment variables)
# 尝试从环境变量获取 API 密钥
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Check if API key is configured
if not OPENROUTER_API_KEY:
    try:
        from dotenv import load_dotenv

        load_dotenv()  # Load environment variables from .env file
        OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
        if not OPENROUTER_API_KEY:
            print(
                "Error: OPENROUTER_API_KEY not found in environment variables or .env file.",
                file=sys.stderr,
            )
            print(
                "Please set the OPENROUTER_API_KEY environment variable or create a .env file with OPENROUTER_API_KEY='YOUR_API_KEY'.",
                file=sys.stderr,
            )
            sys.exit(1)
    except ImportError:
        print(
            "Error: python-dotenv library not found. Please install it with 'pip install python-dotenv'",
            file=sys.stderr,
        )
        print(
            "Alternatively, set the OPENROUTER_API_KEY environment variable directly.",
            file=sys.stderr,
        )
        sys.exit(1)

# Default model for OpenRouter
DEFAULT_MODEL = "deepseek/deepseek-chat"  # 这是一个更常用的 DeepSeek 模型别名


class OpenRouterChat:
    def __init__(self, api_key, model=DEFAULT_MODEL):
        self.api_key = api_key
        self.model = model
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8000",  # 可以根据你的实际应用设置
            "X-Title": "My OpenRouter Chat App",  # 可以根据你的实际应用设置
        }
        # 维持对话历史
        self.messages = [{"role": "system", "content": "你是一个懂中文的友善的IT专家"}]

    def _send_message_to_ai(self, messages_history):
        """
        内部方法：发送消息到AI并处理流式输出。
        返回完整的AI回复字符串，如果发生错误则返回 None。
        """
        payload = {
            "model": self.model,
            "messages": messages_history,  # 使用完整的对话历史
            "stream": True,
        }

        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=self.headers,
                json=payload,
                stream=True,  # 必须设置为 True 以接收流式数据
                timeout=60,  # 设置超时时间，防止长时间无响应
            )

            if response.status_code == 200:
                full_assistant_response = ""  # 用于存储完整的助手回复

                # print(f"Assistant ({self.model}): ", end='', flush=True) # 交互模式下打印前缀

                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode("utf-8")

                        if decoded_line.startswith("data: "):
                            json_str = decoded_line[len("data: ") :].strip()

                            if json_str == "[DONE]":
                                break  # 流结束

                            try:
                                chunk = json.loads(json_str)

                                if "choices" in chunk and len(chunk["choices"]) > 0:
                                    choice = chunk["choices"][0]
                                    if (
                                        "delta" in choice
                                        and "content" in choice["delta"]
                                    ):
                                        content = choice["delta"]["content"]
                                        sys.stdout.write(content)  # 实时打印
                                        sys.stdout.flush()
                                        full_assistant_response += content
                                    elif (
                                        "finish_reason" in choice
                                        and choice["finish_reason"] is not None
                                    ):
                                        pass
                                elif "error" in chunk:
                                    error_data = chunk.get("error", {})
                                    sys.stderr.write(
                                        f"\nError from AI: {error_data.get('message', 'Unknown error')}\n"
                                    )
                                    return None  # 返回None表示出错
                            except json.JSONDecodeError as e:
                                # 忽略无法解析的行
                                continue

                # print() # 交互模式下换行
                return full_assistant_response

            # 处理非 200 状态码的错误
            elif response.status_code == 401:
                sys.stderr.write(
                    "\nAuthentication error (HTTP 401). Please check your API key and try again.\n"
                )
            elif response.status_code == 400:
                try:
                    error_details = response.json()
                    sys.stderr.write(
                        f"\nBad Request (HTTP 400). Details: {error_details.get('message', 'No details provided')}\n"
                    )
                except json.JSONDecodeError:
                    sys.stderr.write(
                        "\nBad Request (HTTP 400). Unable to parse error details.\n"
                    )
            elif response.status_code == 429:
                sys.stderr.write(
                    "\nRate limit exceeded (HTTP 429). Please try again later.\n"
                )
            elif response.status_code in [500, 502, 503, 504]:
                sys.stderr.write(
                    f"\nServer error (HTTP {response.status_code}). Please try again.\n"
                )
            else:
                sys.stderr.write(
                    f"\nUnexpected HTTP error: {response.status_code}. Response: {response.text}\n"
                )
            return None  # 任何非 200 状态码都返回 None

        except requests.exceptions.Timeout:
            sys.stderr.write(
                "\nNetwork Error: The request timed out. Please check your network connection.\n"
            )
            return None
        except requests.exceptions.ConnectionError as e:
            sys.stderr.write(
                f"\nNetwork Error: Could not connect to OpenRouter API. Please check your internet connection. Error: {e}\n"
            )
            return None
        except requests.exceptions.RequestException as e:
            sys.stderr.write(f"An unexpected request error occurred: {str(e)}\n")
            return None
        except json.JSONDecodeError as e:
            sys.stderr.write(
                f"Failed to decode the response JSON (likely malformed data from server): {str(e)}\n"
            )
            return None
        except socket.gaierror as e:
            sys.stderr.write(
                f"\nNetwork Error: DNS resolution failed for 'api.openrouter.ai'. Please check your network connection and DNS settings. Error: {e}\n"
            )
            return None
        except Exception as e:
            sys.stderr.write(f"An unexpected error occurred: {str(e)}\n")
            return None

    def interactive_chat(self):
        """
        交互式聊天模式
        """
        print("Welcome to OpenRouter AI Chat!")
        print(
            "Type your message and press Enter. Type 'exit' to quit. Type 'model <model_name>' to switch models.\n"
        )

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

                print(f"Assistant ({self.model}): ", end="", flush=True)  # 打印前缀
                ai_response_content = self._send_message_to_ai(self.messages)
                print()  # AI回复结束后换行

                if ai_response_content:
                    self.messages.append(
                        {"role": "assistant", "content": ai_response_content}
                    )
                else:
                    # 如果没有有效回复，移除用户刚才的输入，避免对话历史不一致
                    if len(self.messages) > 1 and self.messages[-1]["role"] == "user":
                        self.messages.pop()

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                # 捕获其他未处理的异常，打印并退出，避免循环崩溃
                print(
                    f"\nAn unexpected error occurred in interactive mode: {str(e)}",
                    file=sys.stderr,
                )
                break

    def handle_piped_input(self, piped_content):
        """
        处理管道输入模式
        """
        print(
            f"Processing piped input with model: {self.model}\n", file=sys.stderr
        )  # 打印到标准错误，不影响管道输出

        # 只包含系统消息和当前管道输入作为用户消息
        temp_messages = self.messages[:1]  # 复制系统消息
        temp_messages.append(
            {"role": "user", "content": "请帮我简要总结以下内容"}
        )  # 添加管道内容
        temp_messages.append(
            {"role": "user", "content": piped_content.strip()}
        )  # 添加管道内容

        # 发送请求，不打印 "Assistant: " 前缀，因为通常希望直接输出AI回复
        ai_response_content = self._send_message_to_ai(temp_messages)

        if ai_response_content:
            # 在管道模式下，AI的回复直接就是最终输出，不需要再添加到self.messages
            # 因为管道模式通常是一次性操作，不需要多轮对话历史
            pass
        else:
            sys.stderr.write("Failed to get a response for piped input.\n")
            sys.exit(1)  # 管道模式下，如果失败则退出


if __name__ == "__main__":
    # 再次检查 OPENROUTER_API_KEY，确保它在主执行路径中可用
    if not OPENROUTER_API_KEY:
        print(
            "Please ensure OPENROUTER_API_KEY is set in your environment variables or .env file.",
            file=sys.stderr,
        )
        sys.exit(1)

    chat = OpenRouterChat(OPENROUTER_API_KEY)

    # 检查标准输入是否来自管道或重定向文件
    if not sys.stdin.isatty():
        # 从管道读取所有内容
        piped_data = sys.stdin.read()
        if piped_data:
            chat.handle_piped_input(piped_data)
        else:
            sys.stderr.write("Error: No data piped to stdin.\n")
            sys.exit(1)
    else:
        # 没有管道输入，进入交互式聊天模式
        chat.interactive_chat()
