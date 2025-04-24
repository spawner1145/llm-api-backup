import httpx
import json
import mimetypes
import asyncio
import base64
import os
import uuid
from typing import AsyncGenerator, Dict, List, Optional, Union, Callable
import aiofiles
import logging
from openai import AsyncOpenAI

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenAIAPI:
    def __init__(
        self,
        apikey: str,
        baseurl: str = "https://api.openai.com/v1",
        model: str = "gpt-4o",
        proxy: Optional[str] = None
    ):
        self.apikey = apikey
        self.baseurl = baseurl.rstrip('/')
        self.model = model
        self.client = AsyncOpenAI(
            api_key=apikey,
            base_url=baseurl,
            http_client=httpx.AsyncClient(proxies=proxy, timeout=60.0) if proxy else None
        )

    async def upload_file(self, file_path: str, display_name: Optional[str] = None) -> Dict[str, Union[str, None]]:
        """上传单个文件，使用 client.files.create"""
        try:
            file_size = os.path.getsize(file_path)
            if file_size > 2 * 1024 * 1024 * 1024:  # 2GB 限制
                raise ValueError(f"文件 {file_path} 大小超过 2GB 限制")
        except FileNotFoundError:
            logger.error(f"文件 {file_path} 不存在")
            return {"fileUri": None, "mimeType": None, "error": f"文件 {file_path} 不存在"}

        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type:
            mime_type = "application/octet-stream"
            logger.warning(f"无法检测文件 {file_path} 的 MIME 类型，使用默认值: {mime_type}")

        supported_mime_types = [
            "image/jpeg", "image/png", "image/gif", "image/webp",
            "text/plain", "text/markdown", "application/pdf",
            "audio/mpeg", "audio/wav", "video/mp4"
        ]
        if mime_type not in supported_mime_types:
            logger.warning(f"MIME 类型 {mime_type} 可能不受支持，可能导致处理失败")

        try:
            async with aiofiles.open(file_path, 'rb') as f:
                file_content = await f.read()
                file = await self.client.files.create(
                    file=(display_name or os.path.basename(file_path), file_content, mime_type),
                    purpose="assistants"
                )
                file_uri = file.id
                logger.info(f"文件 {file_path} 上传成功，URI: {file_uri}")
                return {"fileUri": file_uri, "mimeType": mime_type, "error": None}
        except Exception as e:
            logger.error(f"文件 {file_path} 上传失败: {str(e)}")
            return {"fileUri": None, "mimeType": mime_type, "error": str(e)}

    async def upload_files(self, file_paths: List[str], display_names: Optional[List[str]] = None) -> List[Dict[str, Union[str, None]]]:
        """并行上传多个文件"""
        if not file_paths:
            raise ValueError("文件路径列表不能为空")

        if display_names and len(display_names) != len(file_paths):
            raise ValueError("display_names 长度必须与 file_paths 一致")

        tasks = [self.upload_file(file_paths[idx], display_names[idx] if display_names else None) for idx in range(len(file_paths))]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        final_results = []
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"上传文件 {file_paths[idx]} 失败: {str(result)}")
                final_results.append({"fileUri": None, "mimeType": None, "error": str(result)})
            else:
                final_results.append(result)
        return final_results

    async def prepare_inline_data(self, file_path: str) -> Dict[str, Dict[str, str]]:
        """将单个小文件转换为 Base64 编码的 inlineData"""
        file_size = os.path.getsize(file_path)
        if file_size * 4 / 3 > 20 * 1024 * 1024:  # Base64 编码后大小限制 20MB
            raise ValueError(f"文件 {file_path} 过大，无法作为 inlineData，建议使用 File API 上传")

        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type:
            mime_type = "application/octet-stream"
            logger.warning(f"无法检测文件 {file_path} 的 MIME 类型，使用默认值: {mime_type}")

        async with aiofiles.open(file_path, 'rb') as f:
            file_content = await f.read()
        base64_data = base64.b64encode(file_content).decode('utf-8')
        return {
            "inlineData": {
                "mimeType": mime_type,
                "data": base64_data
            }
        }

    async def prepare_inline_data_batch(self, file_paths: List[str]) -> List[Dict[str, Union[Dict[str, str], None]]]:
        """将多个小文件转换为 Base64 编码的 inlineData 列表"""
        if not file_paths:
            raise ValueError("文件路径列表不能为空")

        total_size = 0
        results = []
        for file_path in file_paths:
            try:
                file_size = os.path.getsize(file_path)
                encoded_size = file_size * 4 / 3
                total_size += encoded_size
                if total_size > 20 * 1024 * 1024:
                    raise ValueError(f"文件 {file_path} 加入后总大小超过 20MB，无法作为 inlineData")
                inline_data = await self.prepare_inline_data(file_path)
                results.append(inline_data)
            except Exception as e:
                logger.error(f"处理文件 {file_path} 失败: {str(e)}")
                results.append({"inlineData": None, "error": str(e)})
        return results

    async def _execute_tool(
        self,
        tool_calls: List[Dict],
        tools: Dict[str, Callable]
    ) -> List[Dict]:
        """执行工具调用并返回响应，兼容 OpenAI 和 Gemini 格式"""
        tool_responses = []
        for tool_call in tool_calls:
            name = tool_call.function.name
            if not name:
                logger.error(f"工具调用缺少名称: {tool_call}")
                continue
            # 统一 tool_call_id
            tool_call_id = getattr(tool_call, 'id', None)
            if not tool_call_id:
                tool_call_id = f"call_{uuid.uuid4()}"
                logger.warning(f"工具调用缺少 ID，使用临时 ID: {tool_call_id}")
            args = json.loads(tool_call.function.arguments)
            logger.info(f"执行工具调用: {name}, 参数: {args}, ID: {tool_call_id}")
            func = tools.get(name)
            if func:
                try:
                    if asyncio.iscoroutinefunction(func):
                        result = await func(**args)
                    else:
                        result = func(**args)
                    logger.info(f"工具结果: {name} 返回 {result}, ID: {tool_call_id}")
                    tool_response = {
                        "role": "tool",
                        "content": json.dumps({"result": str(result)}),
                        "tool_call_id": tool_call_id
                    }
                    logger.debug(f"生成工具响应: {json.dumps(tool_response, ensure_ascii=False)}")
                    tool_responses.append((tool_response, tool_call_id))
                except Exception as e:
                    result = f"函数 {name} 执行失败: {str(e)}"
                    logger.error(f"工具错误: {result}, ID: {tool_call_id}")
                    tool_response = {
                        "role": "tool",
                        "content": json.dumps({"error": str(result)}),
                        "tool_call_id": tool_call_id
                    }
                    logger.debug(f"生成工具错误响应: {json.dumps(tool_response, ensure_ascii=False)}")
                    tool_responses.append((tool_response, tool_call_id))
            else:
                logger.error(f"未找到工具: {name}, ID: {tool_call_id}")
                tool_response = {
                    "role": "tool",
                    "content": json.dumps({"error": f"未找到工具 {name}"}),
                    "tool_call_id": tool_call_id
                }
                logger.debug(f"生成工具未找到响应: {json.dumps(tool_response, ensure_ascii=False)}")
                tool_responses.append((tool_response, tool_call_id))
        return tool_responses

    async def _chat_api(
        self,
        messages: List[Dict],
        stream: bool,
        tools: Optional[Dict[str, Callable]] = None,
        max_output_tokens: Optional[int] = None,
        system_instruction: Optional[str] = None,
        topp: Optional[float] = None,
        temperature: Optional[float] = None,
        thinking_budget: Optional[int] = None,
        topk: Optional[int] = None,
        candidate_count: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        response_mime_type: Optional[str] = None,
        response_schema: Optional[Dict] = None,
        seed: Optional[int] = None,
        response_logprobs: Optional[bool] = None,
        logprobs: Optional[int] = None,
        audio_timestamp: Optional[bool] = None,
        safety_settings: Optional[List[Dict]] = None,
        retries: int = 3
    ) -> AsyncGenerator[str, None]:
        """核心 API 调用逻辑，遵循 OpenAI 标准，兼容 Gemini 端点"""
        original_model = self.model

        # 验证参数
        if topp is not None and (topp < 0 or topp > 1):
            raise ValueError("top_p 必须在 0 到 1 之间")
        if temperature is not None and (temperature < 0 or temperature > 2):
            raise ValueError("temperature 必须在 0 到 2 之间")
        if presence_penalty is not None and (presence_penalty < -2 or presence_penalty > 2):
            raise ValueError("presence_penalty 必须在 -2 到 2 之间")
        if frequency_penalty is not None and (frequency_penalty < -2 or frequency_penalty > 2):
            raise ValueError("frequency_penalty 必须在 -2 到 2 之间")
        if logprobs is not None and (logprobs < 0 or logprobs > 20):
            raise ValueError("logprobs 必须在 0 到 20 之间")

        # 构造消息
        api_messages = []
        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")
            if isinstance(content, str):
                content = content
            elif isinstance(content, list):
                content = [
                    part["text"] if "text" in part else
                    f"data:{part['inlineData']['mimeType']};base64,{part['inlineData']['data']}" if "inlineData" in part else
                    part["fileData"] if "fileData" in part else
                    part for part in content
                ]
                content = json.dumps(content)
            api_msg = {
                "role": role,
                "content": content
            }
            if "tool_calls" in msg:
                api_msg["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"]
                        }
                    } for tc in msg["tool_calls"]
                ]
            if "tool_call_id" in msg:
                api_msg["tool_call_id"] = msg["tool_call_id"]
            if "parts" in msg:  # Gemini 格式
                api_msg["parts"] = msg["parts"]
            logger.debug(f"构造消息: {json.dumps(api_msg, ensure_ascii=False)}")
            api_messages.append(api_msg)

        # 构造请求参数，使用 OpenAI 参数名
        request_params = {
            "model": self.model,
            "messages": api_messages,
            "stream": stream
        }
        if max_output_tokens is not None:
            request_params["max_tokens"] = max_output_tokens
        if topp is not None:
            request_params["top_p"] = topp
        if temperature is not None:
            request_params["temperature"] = temperature
        if stop_sequences is not None:
            request_params["stop"] = stop_sequences
        if presence_penalty is not None:
            request_params["presence_penalty"] = presence_penalty
        if frequency_penalty is not None:
            request_params["frequency_penalty"] = frequency_penalty
        if seed is not None:
            request_params["seed"] = seed
        if response_logprobs is not None:
            request_params["logprobs"] = response_logprobs
            if logprobs is not None:
                request_params["top_logprobs"] = logprobs
        if response_mime_type == "application/json" and response_schema is not None:
            request_params["response_format"] = {
                "type": "json_schema",
                "json_schema": {"schema": response_schema, "strict": True}
            }
        elif response_mime_type == "application/json":
            request_params["response_format"] = {"type": "json_object"}

        if tools is not None:
            tool_definitions = []
            for name, func in tools.items():
                params = {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
                if hasattr(func, "__code__"):
                    param_names = func.__code__.co_varnames[:func.__code__.co_argcount]
                    for param in param_names:
                        params["properties"][param] = {"type": "string"}
                        params["required"].append(param)
                else:
                    params["properties"] = {"arg": {"type": "string"}}
                    params["required"] = ["arg"]
                tool_definitions.append({
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": getattr(func, "__doc__", f"调用 {name} 函数"),
                        "parameters": params
                    }
                })
            request_params["tools"] = tool_definitions

        # 打印 POST 请求体
        logger.info(f"发送 POST 请求体: {json.dumps(request_params, ensure_ascii=False, indent=2)}")

        # 执行请求
        if stream:
            tool_calls_buffer = []
            tool_call_ids = {}
            tool_call_args = {}
            async for chunk in await self.client.chat.completions.create(**request_params):
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        yield delta.content
                    elif delta.tool_calls:
                        for tool_call in delta.tool_calls:
                            if tool_call and tool_call.index is not None:
                                # 统一 tool_call_id
                                tool_call_id = tool_call.id if tool_call.id else f"call_{uuid.uuid4()}"
                                tool_call_ids[tool_call.index] = tool_call_id
                                # 初始化工具调用
                                if tool_call.index not in tool_call_args:
                                    tool_call_args[tool_call.index] = {
                                        "name": tool_call.function.name if tool_call.function else "",
                                        "arguments": tool_call.function.arguments or "" if tool_call.function else "",
                                        "id": tool_call_id
                                    }
                                    logger.info(f"工具调用初始化: {tool_call_args[tool_call.index]['name']} (ID: {tool_call_id})")
                                # 累积 arguments
                                if tool_call.function and tool_call.function.arguments:
                                    tool_call_args[tool_call.index]["arguments"] += tool_call.function.arguments
                                    logger.debug(f"参数分片: {tool_call.function.arguments}, ID: {tool_call_id}")
                        # 当流式完成时，构造 tool_calls_buffer
                        if chunk.choices[0].finish_reason == "tool_calls":
                            if tool_call_args:
                                for index, args in tool_call_args.items():
                                    tool_call_id = tool_call_ids.get(index, f"call_{uuid.uuid4()}")
                                    try:
                                        # 验证 arguments 是否为有效 JSON
                                        json.loads(args["arguments"])
                                        tool_calls_buffer.append({
                                            "id": tool_call_id,
                                            "type": "function",
                                            "function": {
                                                "name": args["name"],
                                                "arguments": args["arguments"]
                                            }
                                        })
                                        logger.info(f"参数完成: {args['arguments']}, ID: {tool_call_id}")
                                        logger.info(f"工具调用完成: {args['name']}, ID: {tool_call_id}")
                                    except json.JSONDecodeError:
                                        logger.error(f"工具调用 {args['name']} 的 arguments 无效: {args['arguments']}, ID: {tool_call_id}")
                                        continue
                                if tool_calls_buffer:
                                    # 追加 assistant 消息到 api_messages 和 messages
                                    assistant_message = {
                                        "role": "assistant",
                                        "content": "Calling tools",
                                        "tool_calls": tool_calls_buffer
                                    }
                                    api_messages.append(assistant_message)
                                    messages.append(assistant_message)
                                    logger.debug(f"追加 assistant 消息: {json.dumps(assistant_message, ensure_ascii=False)}")
                                    # 执行工具调用
                                    tool_responses = await self._execute_tool(
                                        [
                                            type('ToolCall', (), {
                                                'function': type('Function', (), {
                                                    'name': tc["function"]["name"],
                                                    'arguments': tc["function"]["arguments"]
                                                })(),
                                                'id': tc["id"]
                                            })() for tc in tool_calls_buffer
                                        ],
                                        tools
                                    )
                                    for tool_response, tool_call_id in tool_responses:
                                        tool_message = {
                                            "role": "tool",
                                            "content": tool_response.get("content", json.dumps(tool_response["parts"][0]["function_response"]["response"])),
                                            "tool_call_id": tool_call_id
                                        }
                                        api_messages.append(tool_message)
                                        messages.append(tool_message)
                                    logger.debug(f"追加工具响应后的消息: {json.dumps(api_messages, ensure_ascii=False, indent=2)}")
                                    # 打印第二次请求的 POST 请求体
                                    second_request_params = request_params.copy()
                                    second_request_params["messages"] = api_messages
                                    second_request_params["stream"] = False
                                    logger.info(f"第二次 POST 请求体: {json.dumps(second_request_params, ensure_ascii=False, indent=2)}")
                                    # 发起第二次请求
                                    async for part in self._chat_api(
                                        api_messages, stream=False, tools=tools,
                                        max_output_tokens=max_output_tokens,
                                        system_instruction=system_instruction,
                                        topp=topp, temperature=temperature,
                                        thinking_budget=thinking_budget,
                                        presence_penalty=presence_penalty,
                                        frequency_penalty=frequency_penalty,
                                        stop_sequences=stop_sequences,
                                        response_mime_type=response_mime_type,
                                        response_schema=response_schema,
                                        seed=seed,
                                        response_logprobs=response_logprobs,
                                        logprobs=logprobs,
                                        retries=retries
                                    ):
                                        yield part
        else:
            for attempt in range(retries):
                try:
                    response = await self.client.chat.completions.create(**request_params)
                    choice = response.choices[0]
                    message = choice.message
                    if message.tool_calls:
                        tool_calls = [
                            {
                                "id": tc.id or f"call_{uuid.uuid4()}",
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments
                                }
                            } for tc in message.tool_calls
                        ]
                        assistant_message = {
                            "role": "assistant",
                            "content": message.content or "Calling tools",
                            "tool_calls": tool_calls
                        }
                        api_messages.append(assistant_message)
                        messages.append(assistant_message)
                        logger.info(f"工具调用: {json.dumps(tool_calls, ensure_ascii=False)}")
                        tool_responses = await self._execute_tool(message.tool_calls, tools)
                        for tool_response, tool_call_id in tool_responses:
                            tool_message = {
                                "role": "tool",
                                "content": tool_response.get("content", json.dumps(tool_response["parts"][0]["function_response"]["response"])),
                                "tool_call_id": tool_call_id
                            }
                            api_messages.append(tool_message)
                            messages.append(tool_message)
                        logger.debug(f"追加工具响应后的消息: {json.dumps(api_messages, ensure_ascii=False, indent=2)}")
                        # 打印第二次请求的 POST 请求体
                        second_request_params = request_params.copy()
                        second_request_params["messages"] = api_messages
                        second_request_params["stream"] = False
                        logger.info(f"第二次 POST 请求体: {json.dumps(second_request_params, ensure_ascii=False, indent=2)}")
                        async for part in self._chat_api(
                            api_messages, stream=False, tools=tools,
                            max_output_tokens=max_output_tokens,
                            system_instruction=system_instruction,
                            topp=topp, temperature=temperature,
                            thinking_budget=thinking_budget,
                            presence_penalty=presence_penalty,
                            frequency_penalty=frequency_penalty,
                            stop_sequences=stop_sequences,
                            response_mime_type=response_mime_type,
                            response_schema=response_schema,
                            seed=seed,
                            response_logprobs=response_logprobs,
                            logprobs=logprobs,
                            retries=retries
                        ):
                            yield part
                    else:
                        if response_logprobs is not None and choice.logprobs:
                            yield f"{message.content or ''}\nLogprobs: {json.dumps(choice.logprobs.content, ensure_ascii=False)}"
                        else:
                            yield message.content or ""
                        if message.content:
                            messages.append({"role": "assistant", "content": message.content})
                    break
                except Exception as e:
                    logger.error(f"API 调用失败 (尝试 {attempt+1}/{retries}): {str(e)}")
                    if attempt == retries - 1:
                        raise
                    await asyncio.sleep(2 ** attempt)

        # 恢复原始模型
        self.model = original_model

    async def chat(
        self,
        messages: Union[str, List[Dict[str, any]]],
        stream: bool = False,
        tools: Optional[Dict[str, Callable]] = None,
        max_output_tokens: Optional[int] = None,
        system_instruction: Optional[str] = None,
        topp: Optional[float] = None,
        temperature: Optional[float] = None,
        thinking_budget: Optional[int] = None,
        topk: Optional[int] = None,
        candidate_count: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        response_mime_type: Optional[str] = None,
        response_schema: Optional[Dict] = None,
        seed: Optional[int] = None,
        response_logprobs: Optional[bool] = None,
        logprobs: Optional[int] = None,
        audio_timestamp: Optional[bool] = None,
        safety_settings: Optional[List[Dict]] = None,
        retries: int = 3
    ) -> AsyncGenerator[str, None]:
        """发起聊天请求，支持多文件上传和多内联内容"""
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
            use_history = False
        else:
            use_history = True

        async for part in self._chat_api(
            messages, stream, tools, max_output_tokens,
            system_instruction, topp, temperature, thinking_budget,
            topk, candidate_count, presence_penalty, frequency_penalty,
            stop_sequences, response_mime_type, response_schema,
            seed, response_logprobs, logprobs, audio_timestamp,
            safety_settings, retries
        ):
            yield part

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.client.aclose()

# 示例工具函数
async def schedule_meeting(start_time: str, duration: str, attendees: str) -> str:
    """安排一个会议，参数包括开始时间、持续时间和与会者"""
    return f"会议已安排：开始时间 {start_time}，持续时间 {duration}，与会者 {attendees}。"

async def get_weather(location: str) -> str:
    """获取指定地点的天气信息"""
    return f"{location} 的天气是晴天，温度 25°C。"

async def get_time(city: str) -> str:
    """获取指定城市的当前时间"""
    return f"{city} 的当前时间是 2025 年 4 月 24 日 13:00。"

async def send_email(to: str, body: str) -> str:
    """发送电子邮件"""
    return f"邮件已发送至 {to}，内容：{body}。"

# 主函数
async def main():
    api = OpenAIAPI(
        apikey="",  # 请替换为实际 Gemini API 密钥
        baseurl="https://generativelanguage.googleapis.com/v1beta/openai/",
        model="gemini-2.0-flash-001"
    )
    tools = {
        "schedule_meeting": schedule_meeting,
        "get_weather": get_weather,
        "get_time": get_time,
        "send_email": send_email
    }

    # 示例 1：单轮对话（非流式，无额外参数）
    print("示例 1：单轮对话（非流式，无额外参数）")
    async for part in api.chat("法国的首都是哪里？", stream=False):
        print(part, end="", flush=True)
    print("\n")

    # 示例 2：多轮对话（非流式，无额外参数）
    print("示例 2：多轮对话（非流式，无额外参数）")
    messages = [
        {"role": "user", "content": "法国的首都是哪里？"},
        {"role": "assistant", "content": "法国的首都是巴黎。"},
        {"role": "user", "content": "巴黎的人口是多少？"}
    ]
    async for part in api.chat(messages, stream=False):
        print(part, end="", flush=True)
    print("\n更新后的消息列表：", json.dumps(messages, ensure_ascii=False, indent=2))
    print()

    # 示例 3：单轮对话（流式，无额外参数）
    print("示例 3：单轮对话（流式，无额外参数）")
    async for part in api.chat("讲一个关于魔法背包的故事。", stream=True):
        print(part, end="", flush=True)
    print("\n")

    # 示例 4：多轮对话（流式，带工具和 presence_penalty）
    print("示例 4：多轮对话（流式，带工具和 presence_penalty）")
    messages = [
        {"role": "user", "content": "今天纽约的天气如何？"}
    ]
    async for part in api.chat(messages, stream=True, tools=tools, presence_penalty=0.5):
        print(part, end="", flush=True)
    print("\n更新后的消息列表：", json.dumps(messages, ensure_ascii=False, indent=2))
    print()

    # 示例 5：多个工具调用（流式，带工具）
    print("示例 5：多个工具调用（流式，带工具）")
    messages = [
        {"role": "user", "content": "请告诉我巴黎和波哥大的天气，并给 Bob 发送一封邮件（bob@email.com），内容为 'Hi Bob'。"}
    ]
    async for part in api.chat(messages, stream=True, tools=tools):
        print(part, end="", flush=True)
    print("\n更新后的消息列表：", json.dumps(messages, ensure_ascii=False, indent=2))
    print()

    # 示例 6：推理模式（非流式，启用推理）
    print("示例 6：推理模式（非流式，启用推理）")
    messages = [
        {"role": "user", "content": "解决数学问题：用数字 10、8、3、7、1 和常用运算符，构造一个表达式等于 24，只能使用每个数字一次。"}
    ]
    async for part in api.chat(messages, stream=False, thinking_budget=24576):
        print("最终回答:", part, end="", flush=True)
    print("\n更新后的消息列表：", json.dumps(messages, ensure_ascii=False, indent=2))
    print()

    # 示例 7：结构化输出（非流式，使用 response_schema）
    print("示例 7：结构化输出（非流式，使用 response_schema）")
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name", "age"]
    }
    messages = [
        {"role": "user", "content": "请提供一个人的信息，包括姓名和年龄。"}
    ]
    async for part in api.chat(
        messages, stream=False, response_mime_type="application/json", response_schema=schema
    ):
        print("结构化输出:", part, end="", flush=True)
    print("\n更新后的消息列表：", json.dumps(messages, ensure_ascii=False, indent=2))
    print()

    # 示例 8：测试 logprobs（非流式，仅限支持的模型，显式传入）
    '''print("示例 8：测试 logprobs（非流式，仅限支持的模型，显式传入）")
    api.model = "gemini-1.5-pro"  # 切换到支持 logprobs 的模型
    messages = [
        {"role": "user", "content": "法国的首都是哪里？"}
    ]
    async for part in api.chat(messages, stream=False, response_logprobs=True, logprobs=5):
        print("带 logprobs 的输出:", part, end="", flush=True)
    print("\n更新后的消息列表：", json.dumps(messages, ensure_ascii=False, indent=2))
    api.model = "gemini-1.5-flash"  # 恢复默认模型
    print()

    # 示例 9：并行使用 File API 上传文件并生成描述（非流式）
    print("示例 9：并行使用 File API 上传文件并生成描述（非流式）")
    try:
        files = ["doc1.txt"]  # 请替换为实际文件路径
        file_data_list = await api.upload_files(files, display_names=["Doc 1"])
        content = [{"type": "text", "text": "描述这些文件的内容（文件已上传，需预处理为文本）。"}]
        for file_data in file_data_list:
            if file_data["fileUri"]:
                content.append({
                    "type": "fileData",
                    "fileData": {"fileUri": file_data["fileUri"], "mimeType": file_data["mimeType"]}
                })
            else:
                print(f"文件上传失败: {file_data['error']}")
        if not any(c["type"] == "fileData" for c in content[1:]):
            print("所有文件上传失败，跳过请求")
        else:
            messages = [{"role": "user", "content": content}]
            async for part in api.chat(messages, stream=False):
                print("文件描述:", part, end="", flush=True)
            print("\n更新后的消息列表：", json.dumps(messages, ensure_ascii=False, indent=2))
    except FileNotFoundError as e:
        print(f"文件不存在: {e}")
    print()

    # 示例 10：使用内联 Base64 文件生成描述（非流式）
    print("示例 10：使用内联 Base64 文件生成描述（非流式）")
    try:
        files = ["image1.jpg"]  # 请替换为实际小文件路径（<10MB）
        inline_data_list = await api.prepare_inline_data_batch(files)
        content = [{"type": "text", "text": "描述这些文件中的内容。"}]
        for inline_data in inline_data_list:
            if inline_data.get("inlineData"):
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{inline_data['inlineData']['mimeType']};base64,{inline_data['inlineData']['data']}"
                    }
                })
            else:
                print(f"内联数据处理失败: {inline_data['error']}")
        messages = [{"role": "user", "content": content}]
        async for part in api.chat(messages, stream=False):
            print("内联文件描述:", part, end="", flush=True)
        print("\n更新后的消息列表：", json.dumps(messages, ensure_ascii=False, indent=2))
    except FileNotFoundError as e:
        print(f"文件不存在: {e}")
    except ValueError as e:
        print(f"内联数据错误: {e}")
    print()'''

if __name__ == "__main__":
    asyncio.run(main())
