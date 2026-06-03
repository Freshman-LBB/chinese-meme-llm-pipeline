import os
import copy
import json
import logging
import hashlib
import itertools
import asyncio
import traceback

import aiofiles
import aiohttp

import jsonlines
from tqdm import tqdm
from types import SimpleNamespace
from functools import cached_property
from importlib.util import find_spec
from sqlitedict import SqliteDict

try:
    import requests
    from aiohttp import ClientSession, TCPConnector, ClientTimeout, ClientResponseError
    from tenacity import RetryError, retry, stop_after_attempt, wait_exponential
    from tqdm.asyncio import tqdm_asyncio
except ModuleNotFoundError:
    pass

# utility class to keep track of json encoded chats
class JsonChatStr:
    prompt: str

    def encode(self, encoding):
        return self.prompt.encode(encoding)


logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)
eval_logger = logging.getLogger("lm-eval")


### SQLite-based caching of LM responses
def hash_args(attr, req, conversation_id=None):
    """Generate hash for both single-turn and multi-turn requests"""
    if conversation_id is not None:
        # Multi-turn case
        dat = json.dumps([attr] + req + [conversation_id])
    else:
        # Single-turn case
        dat = json.dumps([attr] + req)
    return hashlib.sha256(dat.encode("utf-8")).hexdigest()


class CacheHook:
    def __init__(self, cachinglm) -> None:
        if cachinglm is None:
            self.cache = None
            return
        self.cache = cachinglm.cache

    def add_partial(self, attr, req, res, conversation_id=None) -> None:
        if self.cache is None:
            return
        hsh = hash_args(attr, req, conversation_id)
        self.cache[hsh] = res


class CachingLM:
    def __init__(self, lm, cache_db) -> None:
        """LM wrapper that returns cached results if they exist, and uses the underlying LM if not.

        :param lm: LM
            Underlying LM
        :param cache_db: str
            Path to cache db
        """
        self.lm = lm
        self.cache_db = cache_db
        if os.path.dirname(cache_db):
            os.makedirs(os.path.dirname(cache_db), exist_ok=True)
        self.cache = SqliteDict(cache_db, autocommit=True)

        # add hook to lm
        lm.set_cache_hook(self.get_cache_hook())

    def __getattr__(self, attr: str):
        eval_logger.info(f"Wrapping '{attr}' method of LM with caching...")

        def fn(requests, gen_kwargs):
            res = []
            remaining_reqs = []

            eval_logger.info(f"Loading '{attr}' responses from cache where possible...")
            for req in tqdm(requests, desc="Checking cached requests"):
                if isinstance(req, tuple):
                    # Multi-turn case
                    conversation_id, req_content = req
                    hsh = hash_args(attr, req_content, conversation_id)
                else:
                    # Single-turn case
                    hsh = hash_args(attr, req)
                    conversation_id, req_content = None, req

                if hsh in self.cache:
                    ob = self.cache[hsh]
                    assert ob is not None
                    res.append(ob)
                else:
                    res.append(None)
                    remaining_reqs.append((conversation_id, req_content) if conversation_id else req_content)

            eval_logger.info(
                f"Cached requests: {len(requests) - len(remaining_reqs)}, Requests remaining: {len(remaining_reqs)}"
            )

            if remaining_reqs:
                rem_res = getattr(self.lm, attr)(remaining_reqs, gen_kwargs=gen_kwargs)
            else:
                rem_res = []

            # Integrate new results
            resptr = 0
            for req, r in zip(remaining_reqs, rem_res):
                while res[resptr] is not None:
                    resptr += 1
                res[resptr] = r

                if isinstance(req, tuple):
                    # Multi-turn case
                    conversation_id, req_content = req
                    hsh = hash_args(attr, req_content, conversation_id)
                else:
                    # Single-turn case
                    hsh = hash_args(attr, req)
                self.cache[hsh] = r

            self.cache.commit()
            return res

        return fn

    def get_cache_hook(self):
        return CacheHook(self)



def process_stream_output(stream_output):
    reasoning_content = ""  # 定义完整思考过程
    answer_content = ""  # 定义完整回复
    usage = None
    for chunk in stream_output.iter_lines():
        if isinstance(chunk, bytes):
            chunk = chunk.decode('utf-8')

        if not chunk.startswith('data: '):
            continue

        chunk = chunk[6:].strip()
        if chunk == '[DONE]':
            break

        json_obj = json.loads(chunk)
        if not json_obj['choices']:
            usage = json_obj['usage']
        else:
            delta = json_obj['choices'][0]['delta']
            if not delta:
                continue
            if 'reasoning_content' in delta and delta['reasoning_content'] != '' and delta['reasoning_content'] is not None:
                reasoning_content += delta['reasoning_content']
            else:
                if 'content' in delta and delta['content'] != "" and delta['content'] is not None:
                    answer_content += delta['content']
    return answer_content, reasoning_content, usage


class TemplateAPI:
    def __init__(
            self,
            model: str = None,
            base_url: str = None,
            max_retries: int = 10,
            num_concurrent: int = 1,
            max_gen_toks: int = 6144,
            seed: int = 1234,
            api_key: str = "xxx",
    ) -> None:
        missing_packages = [
            pkg
            for pkg in ["aiohttp", "tqdm", "tenacity", "requests"]
            if find_spec(pkg) is None
        ]
        if missing_packages:
            raise ModuleNotFoundError(
                f"Attempted to use an API model, but the required packages {missing_packages} are not installed. "
                'Please install these via `pip install lm-eval[api]` or `pip install -e ."[api]"`'
            )
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self._max_gen_toks = int(max_gen_toks)
        self._seed = int(seed)
        self.max_retries = int(max_retries)
        self._concurrent = int(num_concurrent)

    def set_cache_hook(self, cache_hook) -> None:
        self.cache_hook = cache_hook

    def _create_payload(
            self,
            messages: list,
            gen_kwargs: dict,
    ) -> dict:
        """
        Create the payload to send to the language model API.
        """
        gen_kwargs = gen_kwargs or {}
        gen_kwargs = copy.deepcopy(gen_kwargs)

        # Pop known generation arguments or use defaults
        # temperature = gen_kwargs.pop("temperature", None)

        if isinstance(messages, str):
            key = "prompt"
        else:
            key = "messages"

        payload = {
            key: messages,
            "model": self.model,
            **gen_kwargs,
        }
        temperature = payload.pop('temperature')
        if "o1" in (self.model or "").lower():
            payload.update({
                "max_completion_tokens": 20000
            })
        elif self.model.startswith("ep-20"):
            payload.update({
                "max_tokens": 12288
            })
        elif "claude-3-7-sonnet-20250219#thinking" in (self.model or "").lower():
            if "seed" in payload:
                payload.pop("seed")
        elif self.model == 'qwen-plus-latest#thinking':
            payload['model'] = 'qwen-plus-latest'
            payload.update({
                'enable_thinking': False,
            })
        elif self.model == 'qwen3-8b#thinking':
            payload['model'] = 'qwen3-8b'
            payload.update({
                'enable_thinking': True,
            })
        elif self.model == 'qwen3-14b#thinking':
            payload['model'] = 'qwen3-14b'
            payload.update({
                'enable_thinking': True,
            })
        elif self.model == 'qwen3-32b#thinking':
            payload['model'] = 'qwen3-32b'
            payload.update({
                'enable_thinking': True,
            })
        elif self.model == 'qwen3-235b-a22b#thinking':
            payload['model'] = 'qwen3-235b-a22b'
            payload.update({
                'enable_thinking': True,
            })
        else:
            if 'seed' in gen_kwargs:
                payload.pop("seed")
            payload.update({
                "temperature": temperature,
            })
        return payload

    def parse_generations(self, outputs: list, **kwargs):
        res = []
        if not isinstance(outputs, list):
            outputs = [outputs]

        for out in outputs:
            answer_content, reasoning_content, usage = out['answer_content'], out['reasoning_content'], out['usage']
            res.append([answer_content, reasoning_content, usage])
        return res

    @cached_property
    def header(self) -> dict:
        """Override this property to return the headers for the API request."""
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _handle_inappropriate_content(self, text: str):
        """
        If the server response indicates inappropriate content, return a filtered response.
        """

        if any(
                keyword in text
                for keyword in ["inappropriate content", "high risk", "security reason", "敏感", "BadRequestError"]
        ):
            return {"choices": [{"message": {"content": "content filter"}}]}
        return {}

    def model_call(
            self,
            messages: list,
            gen_kwargs: dict,
    ):
        """
        Synchronous API call.
        """
        payload = self._create_payload(messages, gen_kwargs=gen_kwargs)
        payload['stream'] = True
        payload['stream_options'] = {"include_usage": True}
        # payload['max_tokens'] = 65536
        payload['max_tokens'] = 32768
        try:
            response = requests.post(
                self.base_url,
                json=payload,
                headers=self.header,
            )
            # print(response.text)
            if not response.ok:
                eval_logger.warning(f"API request failed: {response.text}")
                filtered = self._handle_inappropriate_content(str(response.text))
                if filtered:
                    return filtered
            answer_content, reasoning_content, usage = process_stream_output(response)
            response.raise_for_status()
            return {'answer_content': answer_content, 'reasoning_content': reasoning_content, 'usage': usage}

        except RetryError:
            eval_logger.error("API request failed after multiple retries. Check API status.")
            return None

    async def amodel_call(
            self,
            session: ClientSession,
            message: list,
            gen_kwargs: dict,
            cache_keys: tuple,
    ):
        """
        Asynchronous API call.
        """
        payload = self._create_payload(message, gen_kwargs=gen_kwargs)
        payload['stream'] = True
        payload['stream_options'] = {"include_usage": True}
        # if 'qwen' in self.model:
        #     payload['max_tokens'] = 8192
        # elif '4o-mini' in self.model:
        #     payload['max_tokens'] = 16384
        # else:
        #     payload['max_tokens'] = 65536
        # print(payload)
        # print(self.base_url)
        # print(self.header)
        # exit()
        async with session.post(
                url=self.base_url,
                json=payload,
                headers=self.header,
        ) as response:
            if not response.ok:
                error_text = await response.text()
                eval_logger.warning(f"API request failed: {error_text}")
                filtered = self._handle_inappropriate_content(str(error_text))
                outputs = filtered if filtered else None
                if not outputs:
                    response.raise_for_status()
            else:
                reasoning_content = ""  # 定义完整思考过程
                answer_content = ""  # 定义完整回复
                usage = None
                async for chunk in response.content:
                    if isinstance(chunk, bytes):
                        chunk = chunk.decode('utf-8')
                    if not chunk.startswith('data: '):
                        continue
                    chunk = chunk[6:].strip()
                    if chunk == '[DONE]':
                        break
                    json_obj = json.loads(chunk)
                    if 'usage' in json_obj and json_obj['usage'] is not None:
                        usage = json_obj['usage']
                    if not json_obj['choices']:  # 确保choices合法访问
                        continue
                    else:
                        if 'delta' not in json_obj['choices'][0]:  # 确保delta和发访问
                            continue
                        delta = json_obj['choices'][0]['delta']
                        if not delta:
                            continue
                        if 'reasoning_content' in delta and delta['reasoning_content'] is not None and delta['reasoning_content'] != '':
                            reasoning_content += delta['reasoning_content']
                        elif 'reasoning' in delta and delta['reasoning'] is not None and delta['reasoning'] != '':
                            reasoning_content += delta['reasoning']
                        else:
                            if 'content' in delta and delta['content'] != "" and delta['content'] is not None:
                                answer_content += delta['content']

        return answer_content, reasoning_content, usage

    @staticmethod
    def tqdm_update_callback(pbar):
        def callback(future):
            pbar.update(1)  # 每次任务完成时，进度条前进 1

        return callback

    async def get_batched_requests(
            self,
            requests: list,
            gen_kwargs: dict,
    ):
        conn = TCPConnector(limit=self._concurrent)
        async with ClientSession(
            connector=conn, timeout=ClientTimeout(total=6 * 60 * 60),
            trust_env=True
        ) as session:
            retry_ = retry(
                stop=stop_after_attempt(self.max_retries),
                wait=wait_exponential(multiplier=0.5, min=1, max=10),
            )(self.amodel_call)

            if isinstance(requests[0], tuple):
                tasks = [
                    asyncio.create_task(
                        retry_(
                            session=session,
                            message=message,
                            gen_kwargs=gen_kwargs,
                            cache_keys=(message, conversation_id),
                        )
                    )
                    for conversation_id, message in requests
                ]

            else:
                tasks = [
                    asyncio.create_task(
                        retry_(
                            session=session,
                            message=message,
                            gen_kwargs=gen_kwargs,
                            cache_keys=None,
                        )
                    )
                    for message in requests
                ]
            pbar = tqdm(total=len(tasks), desc=f"Requesting API: {self.model}")
            call_back_tasks = []
            for task in tasks:
                task.add_done_callback(self.tqdm_update_callback(pbar))
                call_back_tasks.append(task)
            results = await asyncio.gather(*call_back_tasks, return_exceptions=True)
            pbar.close()
            return results

    def generate_until(
            self,
            requests: list,
            gen_kwargs: dict = None,
            tasks: list = None,
            metadatas: list = None,
            save_file: str = None,
    ):
        res = []
        if self._concurrent == 1:
            pbar = tqdm(desc=f"Requesting API: {self.model}", total=len(requests))
            for idx, req in enumerate(requests):
                # print(req)
                use_cache = False
                if isinstance(req, tuple):
                    conversation_id, req = req
                    use_cache = True
                outputs = retry(
                    stop=stop_after_attempt(self.max_retries),
                    wait=wait_exponential(multiplier=0.5, min=1, max=10),
                    reraise=True,
                )(self.model_call)(req, gen_kwargs)
                if outputs:
                    generated_contents = self.parse_generations(outputs)
                    res.extend(generated_contents)
                    if use_cache:
                        self.cache_hook.add_partial(
                            "generate_until",
                            req,
                            generated_contents,
                            conversation_id
                        )
                    if save_file:
                        with jsonlines.open(save_file, mode='a') as writer:
                            write_target = {
                                'task': tasks[idx],
                                'answer': generated_contents[0][0],
                                'thought': generated_contents[0][1],
                                'usage': generated_contents[0][2],
                                'metadata': metadatas[idx],
                            }
                            writer.write(write_target)
                        print('success save')
                pbar.update(1)
        else:
            results = asyncio.run(
                self.get_batched_requests(
                    requests,
                    gen_kwargs=copy.deepcopy(gen_kwargs)
                )
            )
            res.extend(results)

        res = [item if not isinstance(item, RetryError) else None for item in res]
        return res

    async def async_generate_until(
            self,
            requests: list,
            gen_kwargs: dict = None,
            tasks: list = None,
            metadatas: list = None,
            save_file: str = None,
    ):
        res = []
        if self._concurrent == 1:
            for idx, req in enumerate(requests):
                outputs = await self.async_model_call(req, gen_kwargs)
                if outputs:
                    generated_contents = self.parse_generations(outputs)
                    res.extend(generated_contents)

                    if save_file:
                        async with aiofiles.open(save_file, mode='a') as writer:
                            write_target = {
                                'task': tasks[idx] if tasks else None,
                                'answer': generated_contents[0][0],
                                'thought': generated_contents[0][1],
                                'usage': generated_contents[0][2],
                                'metadata': metadatas[idx] if metadatas else None,
                            }
                            await writer.write(json.dumps(write_target) + '\n')
        else:
            results = await self.get_batched_requests(
                requests,
                gen_kwargs=copy.deepcopy(gen_kwargs)
            )
            res.extend(results)

        res = [item if not isinstance(item, RetryError) else None for item in res]
        return res

    async def async_model_call(
            self,
            message: list,
            gen_kwargs: dict,
    ):
        """
        异步 API 调用
        """
        payload = self._create_payload(message, gen_kwargs=gen_kwargs)
        payload['stream'] = True
        payload['stream_options'] = {"include_usage": True}

        async with aiohttp.ClientSession() as session:
            async with session.post(
                    url=self.base_url,
                    json=payload,
                    headers=self.header,
            ) as response:
                # 类似同步版本的实现，但使用异步方法
                reasoning_content = ""
                answer_content = ""
                usage = None

                async for chunk in response.content:

                    return {'answer_content': answer_content, 'reasoning_content': reasoning_content, 'usage': usage}


class AsyncTemplateAPI:
    def __init__(
            self,
            model: str = None,
            base_url: str = None,
            max_retries: int = 10,
            num_concurrent: int = 1,
            max_gen_toks: int = 6144,
            seed: int = 1234,
            api_key: str = "xxx",
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self._max_gen_toks = int(max_gen_toks)
        self._seed = int(seed)
        self.max_retries = int(max_retries)
        self._concurrent = int(num_concurrent)

    def _create_payload(
            self,
            messages: list,
            gen_kwargs: dict,
    ) -> dict:
        payload = {
            "messages": messages,
            "model": self.model,
            **gen_kwargs,
        }

        temperature = payload.pop('temperature', 0.3)
        payload.update({
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True}
        })

        # 处理 extra_body
        if 'extra_body' in gen_kwargs:
            extra_body = gen_kwargs['extra_body']

            # 创建或使用现有的extra_body
            if 'extra_body' not in payload:
                payload['extra_body'] = {}

            # 针对特定模型添加思考模式
            if ('qwen3' in self.model) and 'enable_thinking' in extra_body:
                payload['extra_body']['enable_thinking'] = extra_body.get('enable_thinking', False)
                payload['extra_body']['thinking_budget'] = extra_body.get('thinking_budget', 500)

            # 添加搜索相关配置
            if 'enable_search' in extra_body:
                payload['extra_body']['enable_search'] = extra_body['enable_search']

                # 如果有搜索选项，也添加进去
                if 'search_options' in extra_body:
                    payload['extra_body']['search_options'] = extra_body['search_options']

        return payload

    @property
    def header(self) -> dict:
        """返回 API 请求的头部信息"""
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def parse_generations(self, outputs: list, **kwargs):
        """解析生成的内容"""
        res = []
        if not isinstance(outputs, list):
            outputs = [outputs]

        for out in outputs:
            # 假设 outputs 是包含内容、推理和使用情况的字典
            answer_content = out.get('content', '*')
            reasoning_content = out.get('reasoning', '*')
            usage = out.get('usage', None)
            res.append([answer_content, reasoning_content, usage])
        return res

    async def async_generate_until(
            self,
            requests: list,
            gen_kwargs: dict = None,
            tasks: list = None,
            metadatas: list = None,
            save_file: str = None,
    ):
        res = []
        for req in requests:
            try:
                outputs = await self.async_model_call(req, gen_kwargs or {})

                if outputs:
                    generated_contents = self.parse_generations([outputs])
                    res.extend(generated_contents)

                    if save_file:
                        async with aiofiles.open(save_file, mode='a') as writer:
                            write_target = {
                                'task': tasks[req] if tasks else None,
                                'answer': generated_contents[0][0],
                                'thought': generated_contents[0][1] or '',  # 尝试获取返回的推理，如果为空则使用空字符串
                                'usage': generated_contents[0][2],
                                'metadata': metadatas[req] if metadatas else None,
                            }
                            await writer.write(json.dumps(write_target) + '\n')
                else:
                    res.append([None, None, None])

            except Exception as e:
                res.append([None, None, None])

        return res

    class ChatCompletions:
        def __init__(self, api_instance):
            self.api = api_instance

        async def create(self, model, messages, extra_body=None, stream=True, stream_options=None,temperature=0.3,top_p=None):
            """模拟OpenAI chat.completions.create接口"""
            gen_kwargs = {
                'temperature': temperature,
            }
            if top_p is not None:
                gen_kwargs['top_p'] = top_p
            if extra_body:
                gen_kwargs['extra_body'] = extra_body

            # 使用现有的异步方法
            result = await self.api.async_model_call(messages, gen_kwargs)

            # 转换为OpenAI格式的流式响应
            if result:
                # 模拟流式chunk格式
                chunks = []

                # 思考内容chunk
                if result.get('reasoning'):
                    reasoning_chunks = self._split_content(result['reasoning'])
                    for chunk_content in reasoning_chunks:
                        chunks.append(self._create_chunk(reasoning_content=chunk_content))

                # 回复内容chunk
                if result.get('content'):
                    content_chunks = self._split_content(result['content'])
                    for chunk_content in content_chunks:
                        chunks.append(self._create_chunk(content=chunk_content))

                # usage chunk
                chunks.append(self._create_chunk(usage=result.get('usage')))

                return AsyncChunkIterator(chunks)

            return AsyncChunkIterator([])

        def _split_content(self, content, chunk_size=10):
            """将内容分割成小块模拟流式输出"""
            return [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]

        def _create_chunk(self, content=None, reasoning_content=None, usage=None):
            """创建模拟的chunk对象"""
            chunk = SimpleNamespace()
            chunk.choices = []

            if content is not None or reasoning_content is not None:
                choice = SimpleNamespace()
                choice.delta = SimpleNamespace()
                if reasoning_content is not None:
                    choice.delta.reasoning_content = reasoning_content
                if content is not None:
                    choice.delta.content = content
                chunk.choices = [choice]
            else:
                chunk.choices = []

            if usage is not None:
                chunk.usage = usage

            return chunk

    @property
    def chat(self):
        """提供chat属性"""
        if not hasattr(self, '_chat'):
            self._chat = SimpleNamespace()
            self._chat.completions = self.ChatCompletions(self)
        return self._chat

    async def async_model_call(
            self,
            message: list,
            gen_kwargs: dict,
    ):
        try:
            # 创建请求载荷
            payload = self._create_payload(message, gen_kwargs)
            payload['stream'] = True
            payload['stream_options'] = {"include_usage": True}

            # 使用 aiohttp 发起请求
            async with aiohttp.ClientSession() as session:
                async with session.post(
                        url=self.base_url,
                        json=payload,
                        headers=self.header,
                ) as response:
                    # 检查响应状态
                    if not response.ok:
                        error_text = await response.text()
                        print(f"API Error: {response.status} - {error_text}")
                        return None

                    # 处理流式响应
                    reasoning_content = ""
                    answer_content = ""
                    usage = None

                    async for chunk in response.content:
                        try:
                            # 解析 chunk
                            chunk_str = chunk.decode('utf-8').strip()

                            if chunk_str.startswith('data: '):
                                chunk_str = chunk_str[6:]

                                if chunk_str == '[DONE]':
                                    break

                                try:
                                    chunk_data = json.loads(chunk_str)
                                except json.JSONDecodeError:
                                    continue

                                # 检查 choices 是否存在且非空
                                if 'choices' in chunk_data and chunk_data['choices']:
                                    delta = chunk_data['choices'][0].get('delta', {})

                                    # 尝试获取推理内容，优先级：reasoning_content > reasoning > content
                                    if 'reasoning_content' in delta and delta['reasoning_content']:
                                        reasoning_content += delta['reasoning_content']
                                    elif 'reasoning' in delta and delta['reasoning']:
                                        reasoning_content += delta['reasoning']
                                    elif 'content' in delta:
                                        answer_content += delta['content']

                                    # 处理 usage
                                    if 'usage' in chunk_data:
                                        usage = chunk_data['usage']

                        except Exception as e:
                            print(f"Error processing chunk: {e}")

                    # 返回处理结果
                    return {
                        'content': answer_content or '',
                        'reasoning': reasoning_content or '',  # 如果没有推理内容，返回空字符串
                        'usage': usage
                    }

        except Exception as e:
            print(f"Async model call error: {e}")
            traceback.print_exc()
            return None


class AsyncChunkIterator:
    """异步chunk迭代器"""

    def __init__(self, chunks):
        self.chunks = chunks
        self.index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index >= len(self.chunks):
            raise StopAsyncIteration
        chunk = self.chunks[self.index]
        self.index += 1
        return chunk