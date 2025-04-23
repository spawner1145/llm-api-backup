import httpx
import json
import mimetypes
import asyncio
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
        baseurl: str = "https://generativelanguage.googleapis.com/v1beta",
        model: str = "gemini-2.0-flash",
        proxy: Optional[str] = None
    ):
        self.apikey = apikey
        self.baseurl = baseurl.rstrip('/')
        self.model = model
        self.client = httpx.AsyncClient(
            base_url=baseurl,
            params={'key': apikey},
            proxies=proxy,
            timeout=60.0
        )

    async def upload_file(self, file_path: str, display_name: Optional[str] = None) -> Dict[str, str]:
        """上传文件到 Gemini API"""
        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type:
            mime_type = "application/octet-stream"

        async with aiofiles.open(file_path, 'rb') as f:
            file_content = await f.read()

        metadata = {
            "file": {
                "displayName": display_name or file_path.split('/')[-1],
                "mimeType": mime_type
            }
        }

        files = {
            "file": (metadata["file"]["displayName"], file_content, mime_type),
            "metadata": (None, json.dumps(metadata), "application/json")
        }

        response = await self.client.post("/upload/files", files=files)
        response.raise_for_status()
        file_data = response.json()

        file_name = file_data["file"]["name"]
        for _ in range(10):
            status_response = await self.client.get(f"/files/{file_name}")
            status_response.raise_for_status()
            status = status_response.json()
            if status["state"] == "ACTIVE":
                return {"fileUri": status["uri"], "mimeType": mime_type}
            elif status["state"] == "FAILED":
                raise ValueError(f"文件处理失败: {status.get('error', '未知错误')}")
            await asyncio.sleep(2)

        raise ValueError("文件处理超时")

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
        # 验证 thinking_budget 和 enable_thinking
        enable_thinking = thinking_budget is not None and thinking_budget > 0
        if thinking_budget is not None:
            if not isinstance(thinking_budget, int) or thinking_budget < 0 or thinking_budget > 24576:
                raise ValueError("thinking_budget 必须是 0 到 24576 之间的整数")
            if self.model not in ["gemini-2.5-flash", "gemini-2.5-pro"]:
                logger.warning(f"模型 {self.model} 不支持 thinking_budget，已忽略该参数")
                thinking_budget = None
                enable_thinking = False

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
        if enable_thinking or thinking_budget is not None:
            generation_config["thinkingConfig"] = {
                "enableThinking": enable_thinking,
                "thinkingBudget": thinking_budget if thinking_budget is not None else 8192
            }

        if generation_config:
            body["generationConfig"] = generation_config

        endpoint = f"/models/{self.model}:{'streamGenerateContent' if stream else 'generateContent'}"
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
                                        elif "thoughts" in part and enable_thinking:
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
                        thoughts = [part["thoughts"] for part in parts if "thoughts" in part and enable_thinking]
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
                    try:
                        logger.error(f"服务器响应: {response.text}")
                    except:
                        logger.error("无法获取服务器响应内容")
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
        """发起聊天请求，支持所有 Gemini 模型参数,见https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/gemini?hl=zh-cn"""
        if isinstance(messages, str):
            messages = [{"role": "user", "parts": [messages]}]
            use_history = False
        else:
            use_history = True

        api_contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            parts = [{"text": p} if isinstance(p, str) else p for p in msg["parts"]]
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
                messages.append({"role": "assistant", "parts": [full_text]})
            logger.info(f"最终消息列表: {json.dumps(messages, ensure_ascii=False, indent=2)}")

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
        {"role": "user", "parts": ["法国的首都是哪里？"]},
        {"role": "assistant", "parts": ["法国的首都是巴黎。"]},
        {"role": "user", "parts": ["巴黎的人口是多少？"]}
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
        {"role": "user", "parts": ["你好！"]},
        {"role": "assistant", "parts": ["你好！有什么可以帮助你的？"]},
        {"role": "user", "parts": ["今天纽约的天气如何？"]}
    ]
    async for part in api.chat(messages, stream=True, tools=tools):
        print(part, end="", flush=True)
    print("\n更新后的消息列表：", messages)
    print()

    # 示例 5：多个工具调用（非流式）
    print("示例 5：多个工具调用（非流式）")
    messages = [
        {"role": "user", "parts": ["请安排一个明天上午10点的会议，持续1小时，与会者是Alice和Bob。然后告诉我纽约的天气和时间。"]}
    ]
    async for part in api.chat(messages, stream=False, tools=tools):
        print(part)
    print("更新后的消息列表：", messages)
    print()

    # 示例 6：思考模式（非流式，启用思考）
    print("示例 6：思考模式（非流式，启用思考）")
    messages = [
        {"role": "user", "parts": ["解决数学问题：用数字 10、8、3、7、1 和常用运算符，构造一个表达式等于 24，只能使用每个数字一次。"]}
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
        {"role": "user", "parts": ["解决数学问题：用数字 10、8、3、7、1 和常用运算符，构造一个表达式等于 24，只能使用每个数字一次。"]}
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
        {"role": "user", "parts": ["解决数学问题：用数字 10、8、3、7、1 和常用运算符，构造一个表达式等于 24，只能使用每个数字一次。"]}
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
        {"role": "user", "parts": ["请提供一个人的信息，包括姓名和年龄。"]}
    ]
    async for part in api.chat(
        messages, stream=False, response_mime_type="application/json", response_schema=schema
    ):
        print("结构化输出:", part)
    print("更新后的消息列表：", messages)
    print()

if __name__ == "__main__":
    asyncio.run(main())
