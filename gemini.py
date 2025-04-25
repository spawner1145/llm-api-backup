import httpx
import json
import mimetypes
import asyncio
import base64
import os
from typing import AsyncGenerator, Dict, List, Optional, Union, Callable
import aiofiles
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiAPI:
    def __init__(
        self,
        apikey: str,
        baseurl: str = "https://generativelanguage.googleapis.com",
        model: str = "gemini-2.0-flash-001",  # 更新为支持多模态的模型
        proxies: Optional[Dict[str, str]] = None
    ):
        self.apikey = apikey
        self.baseurl = baseurl.rstrip('/')
        self.model = model
        self.client = httpx.AsyncClient(
            base_url=baseurl,
            params={'key': apikey},
            proxies=proxies,
            timeout=60.0
        )

    async def upload_file(self, file_path: str, display_name: Optional[str] = None) -> Dict[str, Union[str, None]]:
        """上传单个文件到 Gemini File API，并检查 ACTIVE 状态"""
        # 验证文件大小
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

        # 检查支持的 MIME 类型
        supported_mime_types = [
            "image/jpeg", "image/png", "image/gif", "image/webp",
            "video/mp4", "video/mpeg", "video/avi", "video/wmv", "video/flv",
            "audio/mp3", "audio/wav", "audio/ogg", "audio/flac",
            "text/plain", "text/markdown"
        ]
        if mime_type not in supported_mime_types:
            logger.warning(f"MIME 类型 {mime_type} 可能不受支持，可能导致上传失败")

        # 上传文件
        try:
            async with aiofiles.open(file_path, 'rb') as f:
                files = {'file': (display_name or os.path.basename(file_path), await f.read(), mime_type)}
                logger.info(f"上传文件请求: {file_path}")
                response = await self.client.post("/upload/v1beta/files", files=files)
                response.raise_for_status()
                file_data = response.json()
                logger.info(f"文件上传响应: {file_data}")
                file_uri = file_data['file'].get('uri')
                if not file_uri:
                    logger.error(f"文件 {file_path} 上传成功但未返回 URI: {file_data}")
                    return {"fileUri": None, "mimeType": mime_type, "error": f"文件 {file_path} 上传成功但未返回 URI"}
        except httpx.HTTPStatusError as e:
            logger.error(f"文件 {file_path} 上传失败: {e.response.text}")
            return {"fileUri": None, "mimeType": mime_type, "error": f"文件 {file_path} 上传失败: {e.response.text}"}
        except Exception as e:
            logger.error(f"文件 {file_path} 上传失败: {str(e)}")
            return {"fileUri": None, "mimeType": mime_type, "error": f"文件 {file_path} 上传失败: {str(e)}"}

        # 等待文件状态变为 ACTIVE
        if not await self.wait_for_file_active(file_uri, timeout=120, interval=2):
            logger.error(f"文件 {file_path} 未能在规定时间内变为 ACTIVE 状态")
            return {"fileUri": None, "mimeType": mime_type, "error": f"文件 {file_path} 未能在规定时间内变为 ACTIVE 状态"}

        logger.info(f"文件 {file_path} 上传并激活成功，URI: {file_uri}")
        return {"fileUri": file_uri, "mimeType": mime_type, "error": None}

    async def wait_for_file_active(self, file_uri: str, timeout: Optional[int] = None, interval: int = 2) -> bool:
        """等待文件状态变为 ACTIVE"""
        file_id = file_uri.split('/')[-1]  # 提取文件 ID
        start_time = asyncio.get_event_loop().time()

        while timeout is None or (asyncio.get_event_loop().time() - start_time < timeout):
            try:
                response = await self.client.get(f"/v1beta/files/{file_id}")
                response.raise_for_status()
                file_info = response.json()
                logger.debug(f"文件 {file_id} 状态响应: {file_info}")
                state = file_info.get('state')
                if state == "ACTIVE":
                    logger.info(f"文件 {file_id} 已就绪，状态: {state}")
                    return True
                elif state == "FAILED":
                    logger.error(f"文件 {file_id} 处理失败，状态: {state}")
                    return False
                else:
                    logger.info(f"文件 {file_id} 仍在处理中，状态: {state}")
                    await asyncio.sleep(interval)
            except httpx.HTTPStatusError as e:
                logger.error(f"检查文件 {file_id} 状态失败: {e.response.text}")
                return False
            except Exception as e:
                logger.error(f"检查文件 {file_id} 状态失败: {str(e)}")
                return False
        logger.error(f"等待文件 {file_id} 超时 ({timeout}秒)")
        return False

    async def upload_files(self, file_paths: List[str], display_names: Optional[List[str]] = None) -> List[Dict[str, Union[str, None]]]:
        """并行上传多个文件到 Gemini File API"""
        if not file_paths:
            raise ValueError("文件路径列表不能为空")

        if display_names and len(display_names) != len(file_paths):
            raise ValueError("display_names 长度必须与 file_paths 一致")

        # 创建并行上传任务
        tasks = []
        for idx, file_path in enumerate(file_paths):
            display_name = display_names[idx] if display_names else None
            tasks.append(self.upload_file(file_path, display_name))

        # 并行执行上传
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
        # Base64 编码后大小增加约 33%，检查是否超 20MB
        if file_size * 4 / 3 > 20 * 1024 * 1024:
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
                encoded_size = file_size * 4 / 3  # Base64 编码后大小
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
        function_calls: List[Dict],
        tools: Dict[str, Callable]
    ) -> List[Dict]:
        """执行工具调用并返回响应"""
        function_responses = []
        for function_call in function_calls:
            name = function_call["name"]
            args = function_call.get("args", {})
            logger.info(f"执行工具调用: {name}, 参数: {args}")
            func = tools.get(name)
            if func:
                try:
                    if asyncio.iscoroutinefunction(func):
                        result = await func(**args)
                    else:
                        result = func(**args)
                except Exception as e:
                    result = f"函数 {name} 执行失败: {str(e)}"
                function_responses.append({
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": name,
                                "response": {"result": str(result)}
                            }
                        }
                    ]
                })
        return function_responses

    async def _chat_api(
        self,
        api_contents: List[Dict],
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
    ) -> AsyncGenerator[Union[str, Dict], None]:
        """核心 API 调用逻辑，支持所有 Gemini 模型参数"""
        if thinking_budget is not None:
            if not isinstance(thinking_budget, int) or thinking_budget < 0 or thinking_budget > 24576:
                raise ValueError("thinking_budget 必须是 0 到 24576 之间的整数")

        # 验证其他参数
        if topp is not None and (topp < 0 or topp > 1):
            raise ValueError("topP 必须在 0 到 1 之间")
        if temperature is not None and (temperature < 0 or temperature > 2):
            raise ValueError("temperature 必须在 0 到 2 之间")
        if topk is not None and topk < 1:
            raise ValueError("topK 必须是正整数")
        if candidate_count is not None and (candidate_count < 1 or candidate_count > 8):
            raise ValueError("candidateCount 必须在 1 到 8 之间")
        if presence_penalty is not None and (presence_penalty < -2 or presence_penalty > 2):
            raise ValueError("presencePenalty 必须在 -2 到 2 之间")
        if frequency_penalty is not None and (frequency_penalty < -2 or frequency_penalty > 2):
            raise ValueError("frequencyPenalty 必须在 -2 到 2 之间")
        if response_mime_type is not None and response_mime_type not in ["text/plain", "application/json"]:
            raise ValueError("responseMimeType 必须是 'text/plain' 或 'application/json'")
        if logprobs is not None and (logprobs < 0 or logprobs > 5):
            raise ValueError("logprobs 必须在 0 到 5 之间")

        body = {"contents": api_contents}
        if tools:
            function_declarations = []
            for name, func in tools.items():
                params = []
                if hasattr(func, "__code__"):
                    params = func.__code__.co_varnames[:func.__code__.co_argcount]
                else:
                    logger.warning(f"函数 {name} 没有 __code__ 属性，使用空参数列表")
                parameters = {
                    "type": "object",
                    "properties": {param: {"type": "string"} for param in params},
                    "required": params
                }
                function_declarations.append({
                    "name": name,
                    "description": getattr(func, "__doc__", f"调用 {name} 函数"),
                    "parameters": parameters
                })
            body["tools"] = [{"functionDeclarations": function_declarations}]
        if system_instruction:
            body["systemInstruction"] = {"parts": [{"text": system_instruction}]}
        if safety_settings:
            body["safetySettings"] = safety_settings

        generation_config = {}
        if max_output_tokens:
            generation_config["maxOutputTokens"] = max_output_tokens
        if topp:
            generation_config["topP"] = topp
        if temperature:
            generation_config["temperature"] = temperature
        if topk:
            generation_config["topK"] = topk
        if candidate_count:
            generation_config["candidateCount"] = candidate_count
        if presence_penalty:
            generation_config["presencePenalty"] = presence_penalty
        if frequency_penalty:
            generation_config["frequencyPenalty"] = frequency_penalty
        if stop_sequences:
            generation_config["stopSequences"] = stop_sequences
        if response_mime_type:
            generation_config["responseMimeType"] = response_mime_type
        if response_schema:
            generation_config["responseSchema"] = response_schema
        if seed:
            generation_config["seed"] = seed
        if response_logprobs is not None:
            generation_config["responseLogprobs"] = response_logprobs
        if logprobs is not None:
            generation_config["logprobs"] = logprobs
        if audio_timestamp is not None:
            generation_config["audioTimestamp"] = audio_timestamp
        if thinking_budget is not None:
            generation_config["thinkingConfig"] = {
                "thinkingBudget": thinking_budget if thinking_budget is not None else 8192
            }

        if generation_config:
            body["generationConfig"] = generation_config

        endpoint = f"/v1beta/models/{self.model}:{'streamGenerateContent' if stream else 'generateContent'}"
        logger.info(f"请求端点: {endpoint}")

        if stream:
            async with self.client.stream("POST", endpoint, json=body, params={'alt': 'sse'}) as response:
                logger.info(f"流式响应状态: {response.status_code}")
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as e:
                    logger.error(f"流式 HTTP 错误: {e}")
                    logger.error(f"失败的请求体: {json.dumps(body, ensure_ascii=False, indent=2)}")
                    logger.error(f"服务器响应: {await response.aread()}")
                    raise
                model_message = {"role": "model", "parts": []}
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[len("data: "):].strip()
                        if data:
                            try:
                                chunk = json.loads(data)
                                for candidate in chunk.get("candidates", []):
                                    for part in candidate.get("content", {}).get("parts", []):
                                        if "text" in part:
                                            model_message["parts"].append({"text": part["text"]})
                                            yield part["text"]
                                        elif "thoughts" in part:
                                            model_message["parts"].append({"thoughts": part["thoughts"]})
                                            yield {"thoughts": part["thoughts"]}
                                        elif "functionCall" in part and tools:
                                            model_message["parts"].append({"functionCall": part["functionCall"]})
                                            function_calls = [part["functionCall"]]
                                            function_responses = await self._execute_tool(function_calls, tools)
                                            api_contents.append(model_message)
                                            api_contents.extend(function_responses)
                                            async for text in self._chat_api(
                                                api_contents, stream=False, tools=tools,
                                                max_output_tokens=max_output_tokens,
                                                system_instruction=system_instruction,
                                                topp=topp, temperature=temperature,
                                                thinking_budget=thinking_budget,
                                                topk=topk, candidate_count=candidate_count,
                                                presence_penalty=presence_penalty,
                                                frequency_penalty=frequency_penalty,
                                                stop_sequences=stop_sequences,
                                                response_mime_type=response_mime_type,
                                                response_schema=response_schema,
                                                seed=seed, response_logprobs=response_logprobs,
                                                logprobs=logprobs, audio_timestamp=audio_timestamp,
                                                safety_settings=safety_settings,
                                                retries=retries
                                            ):
                                                yield text
                            except json.JSONDecodeError as e:
                                logger.error(f"流式 JSON 解析错误: {e}")
                if model_message["parts"] and not any("functionCall" in part for part in model_message["parts"]):
                    api_contents.append(model_message)
        else:
            for attempt in range(retries):
                try:
                    response = await self.client.post(endpoint, json=body)
                    logger.info(f"非流式响应状态: {response.status_code}")
                    response.raise_for_status()
                    result = response.json()
                    candidate = result["candidates"][0]
                    model_message = {
                        "role": candidate["content"]["role"],
                        "parts": candidate["content"]["parts"]
                    }
                    api_contents.append(model_message)
                    function_calls = [part["functionCall"] for part in candidate["content"]["parts"] if "functionCall" in part]
                    if function_calls:
                        logger.info(f"发现函数调用: {function_calls}")
                        function_responses = await self._execute_tool(function_calls, tools)
                        api_contents.extend(function_responses)
                        async for text in self._chat_api(
                            api_contents, stream=False, tools=tools,
                            max_output_tokens=max_output_tokens,
                            system_instruction=system_instruction,
                            topp=topp, temperature=temperature,
                            thinking_budget=thinking_budget,
                            topk=topk, candidate_count=candidate_count,
                            presence_penalty=presence_penalty,
                            frequency_penalty=frequency_penalty,
                            stop_sequences=stop_sequences,
                            response_mime_type=response_mime_type,
                            response_schema=response_schema,
                            seed=seed, response_logprobs=response_logprobs,
                            logprobs=logprobs, audio_timestamp=audio_timestamp,
                            safety_settings=safety_settings,
                            retries=retries
                        ):
                            yield text
                    else:
                        parts = candidate["content"]["parts"]
                        thoughts = [part["thoughts"] for part in parts if "thoughts" in part]
                        text = "".join(part["text"] for part in parts if "text" in part)
                        logprobs_data = candidate.get("logprobs", []) if response_logprobs else []
                        if thoughts or logprobs_data:
                            yield {
                                "thoughts": thoughts if thoughts else None,
                                "text": text,
                                "logprobs": logprobs_data
                            }
                        else:
                            yield text
                    break
                except httpx.HTTPStatusError as e:
                    logger.error(f"HTTP 错误 (尝试 {attempt+1}/{retries}): {e}")
                    logger.error(f"失败的请求体: {json.dumps(body, ensure_ascii=False, indent=2)}")
                    logger.error(f"服务器响应: {e.response.text}")
                    if attempt == retries - 1:
                        raise
                    await asyncio.sleep(2 ** attempt)
                except json.JSONDecodeError:
                    logger.error(f"JSON 解析错误 (尝试 {attempt+1}/{retries})")
                    logger.error(f"失败的请求体: {json.dumps(body, ensure_ascii=False, indent=2)}")
                    if attempt == retries - 1:
                        raise ValueError("无效的 JSON 响应")
                    await asyncio.sleep(2 ** attempt)

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
    ) -> AsyncGenerator[Union[str, Dict, List[Dict[str, any]]], None]:
        """发起聊天请求，支持多文件上传和多内联内容"""
        if isinstance(messages, str):
            messages = [{"role": "user", "parts": [{"text": messages}]}]
            use_history = False
        else:
            use_history = True

        api_contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            parts = []
            for p in msg["parts"]:
                if isinstance(p, str):
                    parts.append({"text": p})
                elif isinstance(p, dict) and ("fileData" in p or "inlineData" in p):
                    parts.append(p)
                else:
                    parts.append(p)
            api_contents.append({"role": role, "parts": parts})

        #logger.info(f"初始 API contents: {json.dumps(api_contents, ensure_ascii=False, indent=2)}")

        full_text = ""
        thoughts = []
        logprobs_data = []
        async for part in self._chat_api(
            api_contents, stream, tools, max_output_tokens,
            system_instruction, topp, temperature, thinking_budget,
            topk, candidate_count, presence_penalty, frequency_penalty,
            stop_sequences, response_mime_type, response_schema,
            seed, response_logprobs, logprobs, audio_timestamp,
            safety_settings, retries
        ):
            if isinstance(part, dict):
                if "thoughts" in part and part["thoughts"]:
                    thoughts.extend(part["thoughts"] if isinstance(part["thoughts"], list) else [part["thoughts"]])
                if "logprobs" in part and part["logprobs"]:
                    logprobs_data.extend(part["logprobs"])
                full_text += part["text"]
                yield part
            else:
                full_text += part
                yield part

        if use_history and full_text:
            if thoughts or logprobs_data:
                parts = []
                if thoughts:
                    parts.append({"thoughts": thoughts})
                parts.append({"text": full_text})
                if logprobs_data:
                    parts.append({"logprobs": logprobs_data})
                messages.append({"role": "assistant", "parts": parts})
            else:
                messages.append({"role": "assistant", "parts": [{"text": full_text}]})
            #logger.info(f"最终消息列表: {json.dumps(messages, ensure_ascii=False, indent=2)}")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.client.aclose()

# 示例工具函数
async def schedule_meeting(start_time: str, duration: str, attendees: str) -> str:
    """安排一个会议，参数包括开始时间、持续时间和与会者"""
    return f"会议已安排：开始时间 {start_time}，持续时间 {duration}，与会者 {attendees}。"

async def get_weather(location: str) -> str:
    """获取指定地点的天气信息（示例）"""
    return f"{location} 的天气是晴天，温度 25°C。"

async def get_time(city: str) -> str:
    """获取指定城市的当前时间（示例）"""
    return f"{city} 的当前时间是 2025 年 4 月 23 日 18:00。"

# 主函数
async def main():
    api = GeminiAPI(apikey="")  # 请替换为你的实际 API 密钥
    tools = {
        "schedule_meeting": schedule_meeting,
        "get_weather": get_weather,
        "get_time": get_time
    }

    # 示例 1：单轮对话（非流式）
    print("示例 1：单轮对话（非流式）")
    async for part in api.chat("法国的首都是哪里？", stream=False):
        print(part)
    print()

    # 示例 2：多轮对话（非流式）
    print("示例 2：多轮对话（非流式）")
    messages = [
        {"role": "user", "parts": [{"text": "法国的首都是哪里？"}]},
        {"role": "assistant", "parts": [{"text": "法国的首都是巴黎。"}]},
        {"role": "user", "parts": [{"text": "巴黎的人口是多少？"}]}
    ]
    async for part in api.chat(messages, stream=False):
        print(part)
    print("更新后的消息列表：", messages)
    print()

    # 示例 3：单轮对话（流式）
    print("示例 3：单轮对话（流式）")
    async for part in api.chat("讲一个关于魔法背包的故事。", stream=True):
        print(part, end="", flush=True)
    print("\n")

    # 示例 4：多轮对话（流式，带工具）
    print("示例 4：多轮对话（流式，带工具）")
    messages = [
        {"role": "user", "parts": [{"text": "你好！"}]},
        {"role": "assistant", "parts": [{"text": "你好！有什么可以帮助你的？"}]},
        {"role": "user", "parts": [{"text": "今天纽约的天气如何？"}]}
    ]
    async for part in api.chat(messages, stream=True, tools=tools):
        print(part, end="", flush=True)
    print("\n更新后的消息列表：", messages)
    print()

    # 示例 5：多个工具调用（非流式）
    print("示例 5：多个工具调用（非流式）")
    messages = [
        {"role": "user", "parts": [{"text": "请安排一个明天上午10点的会议，持续1小时，与会者是Alice和Bob。然后告诉我纽约的天气和时间。"}]}
    ]
    async for part in api.chat(messages, stream=False, tools=tools):
        print(part)
    print("更新后的消息列表：", messages)
    print()

    # 示例 6：思考模式（非流式，启用思考）
    print("示例 6：思考模式（非流式，启用思考）") # 注意，只有2.5系的模型才支持思考，别的模型请勿传入thinking_budget参数
    messages = [
        {"role": "user", "parts": [{"text": "解决数学问题：用数字 10、8、3、7、1 和常用运算符，构造一个表达式等于 24，只能使用每个数字一次。"}]}
    ]
    async for part in api.chat(messages, stream=False, thinking_budget=24576):
        if isinstance(part, dict):
            print("思考过程:", part["thoughts"])
            print("最终回答:", part["text"])
            if part["logprobs"]:
                print("Logprobs:", part["logprobs"])
        else:
            print(part)
    print("更新后的消息列表：", messages)
    print()

    # 示例 7：思考模式（流式，启用思考）
    print("示例 7：思考模式（流式，启用思考）")
    messages = [
        {"role": "user", "parts": [{"text": "解决数学问题：用数字 10、8、3、7、1 和常用运算符，构造一个表达式等于 24，只能使用每个数字一次。"}]}
    ]
    async for part in api.chat(messages, stream=True, thinking_budget=24576):
        if isinstance(part, dict) and "thoughts" in part:
            print("思考过程:", part["thoughts"])
        else:
            print("流式输出:", part, end="", flush=True)
    print("\n更新后的消息列表：", messages)
    print()

    # 示例 8：思考模式（非流式，禁用思考）
    print("示例 8：思考模式（非流式，禁用思考）")
    messages = [
        {"role": "user", "parts": [{"text": "解决数学问题：用数字 10、8、3、7、1 和常用运算符，构造一个表达式等于 24，只能使用每个数字一次。"}]}
    ]
    async for part in api.chat(messages, stream=False, thinking_budget=0):
        print("最终回答:", part)
    print("更新后的消息列表：", messages)
    print()

    # 示例 9：结构化输出（非流式，使用 response_schema）
    print("示例 9：结构化输出（非流式，使用 response_schema）")
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name", "age"]
    }
    messages = [
        {"role": "user", "parts": [{"text": "请提供一个人的信息，包括姓名和年龄。"}]}
    ]
    async for part in api.chat(
        messages, stream=False, response_mime_type="application/json", response_schema=schema
    ):
        print("结构化输出:", part)
    print("更新后的消息列表：", messages)
    print()

    # 示例 11：并行使用 File API 上传多个图像并生成描述（非流式）
    print("示例 11：并行使用 File API 上传多个图像并生成描述（非流式）")
    try:
        image_files = ["《Break the Cocoon》封面.jpg", "ComfyUI_temp_poxuo_00001_.png"]  # 请替换为实际图像路径
        file_data_list = await api.upload_files(image_files, display_names=["Image 1", "Image 2"])
        parts = [{"text": "描述这些图片中的内容。"}]
        for file_data in file_data_list:
            if file_data["fileUri"]:
                parts.append({"fileData": {"fileUri": file_data["fileUri"], "mimeType": file_data["mimeType"]}})
            else:
                print(f"文件上传失败: {file_data['error']}")
        if not any("fileData" in part for part in parts[1:]):
            print("所有文件上传失败，跳过请求")
        else:
            messages = [{"role": "user", "parts": parts}]
            async for part in api.chat(messages, stream=False):
                print("多图像描述:", part)
            print("更新后的消息列表：", messages)
    except FileNotFoundError as e:
        print(f"文件不存在: {e}")
    print()

    # 示例 12：并行使用 File API 上传图像和视频并总结内容（非流式）
    print("示例 12：并行使用 File API 上传图像和视频并总结内容（非流式）")
    try:
        media_files = ["《Break the Cocoon》封面.jpg", "QQ2025418-133636.mp4"]  # 请替换为实际文件路径
        file_data_list = await api.upload_files(media_files, display_names=["Image 1", "Video 1"])
        parts = [{"text": "总结这些媒体文件的内容。"}]
        for file_data in file_data_list:
            if file_data["fileUri"]:
                parts.append({"fileData": {"fileUri": file_data["fileUri"], "mimeType": file_data["mimeType"]}})
            else:
                print(f"文件上传失败: {file_data['error']}")
        if not any("fileData" in part for part in parts[1:]):
            print("所有文件上传失败，跳过请求")
        else:
            messages = [{"role": "user", "parts": parts}]
            async for part in api.chat(messages, stream=False):
                print("媒体总结:", part)
            print("更新后的消息列表：", messages)
    except FileNotFoundError as e:
        print(f"文件不存在: {e}")
    print()

    # 示例 13：使用多个内联 Base64 图像生成描述（非流式）
    print("示例 13：使用多个内联 Base64 图像生成描述（非流式）")
    try:
        image_files = ["《Break the Cocoon》封面.jpg", "ComfyUI_temp_poxuo_00001_.png"]  # 请替换为实际小图像路径（<10MB）
        inline_data_list = await api.prepare_inline_data_batch(image_files)
        parts = [{"text": "描述这些图片中的内容。"}]
        for inline_data in inline_data_list:
            if inline_data.get("inlineData"):
                parts.append(inline_data)
            else:
                print(f"内联数据处理失败: {inline_data['error']}")
        messages = [{"role": "user", "parts": parts}]
        async for part in api.chat(messages, stream=False):
            print("多内联图像描述:", part)
        print("更新后的消息列表：", messages)
    except FileNotFoundError as e:
        print(f"文件不存在: {e}")
    except ValueError as e:
        print(f"内联数据错误: {e}")
    print()

    # 示例 14：混合并行上传文件和多内联数据（文本文件 + 多图像，非流式）
    print("示例 14：混合并行上传文件和多内联数据（文本文件 + 多图像，非流式）")
    try:
        text_files = ["requirements.txt"]  # 请替换为实际文本文件路径
        image_files = ["《Break the Cocoon》封面.jpg", "ComfyUI_temp_poxuo_00001_.png"]  # 请替换为实际小图像路径（<10MB）
        file_data_list = await api.upload_files(text_files, display_names=["Sample Text"])
        inline_data_list = await api.prepare_inline_data_batch(image_files)
        parts = [{"text": "根据上传的文本文件总结内容，并描述旁边的图片。"}]
        for file_data in file_data_list:
            if file_data["fileUri"]:
                parts.append({"fileData": {"fileUri": file_data["fileUri"], "mimeType": file_data["mimeType"]}})
            else:
                print(f"文件上传失败: {file_data['error']}")
        for inline_data in inline_data_list:
            if inline_data.get("inlineData"):
                parts.append(inline_data)
            else:
                print(f"内联数据处理失败: {inline_data['error']}")
        if not any("fileData" in part or "inlineData" in part for part in parts[1:]):
            print("所有文件和内联数据处理失败，跳过请求")
        else:
            messages = [{"role": "user", "parts": parts}]
            async for part in api.chat(messages, stream=False):
                print("混合内容输出:", part)
            print("更新后的消息列表：", messages)
    except FileNotFoundError as e:
        print(f"文件不存在: {e}")
    except ValueError as e:
        print(f"内联数据错误: {e}")
    print()

if __name__ == "__main__":
    asyncio.run(main())
