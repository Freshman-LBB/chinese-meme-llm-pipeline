import traceback

from tqdm import tqdm
import asyncio
import re
import os
import json

from llm_api.template_api import TemplateAPI, AsyncTemplateAPI


def process_single_request(question, client, model):
    reasoning_content = ""  # 定义完整思考过程
    answer_content = ""  # 定义完整回复
    stream = client.chat.completions.create(
        model=model,
        messages=[{
            'role': 'user',
            'content': question
        }],
        stream=True,
    )
    for chunk in stream:
        # 如果chunk.choices为空，则打印usage
        if not chunk.choices:
            print("\nUsage:")
            print(chunk.usage)
        else:
            delta = chunk.choices[0].delta
            # 思考过程
            if hasattr(delta, 'reasoning_content') and delta.reasoning_content != None:
                reasoning_content += delta.reasoning_content
            else:
                # 回复
                if delta.content != None:
                    answer_content += delta.content
    return answer_content, reasoning_content


async def call_client(question, client, model):
    reasoning_content = ""  # 定义完整思考过程
    answer_content = ""  # 定义完整回复
    is_answering = False  # 判断是否结束思考过程并开始回复
    stream = await client.chat.completions.create(
        model=model,
        messages=[{
            'role': 'user',
            'content': question
        }],
        stream=True,
    )
    async for chunk in stream:
        # 如果chunk.choices为空，则打印usage
        if not chunk.choices:
            print("\nUsage:")
            print(chunk.usage)
        else:
            delta = chunk.choices[0].delta
            # 打印思考过程
            if hasattr(delta, 'reasoning_content') and delta.reasoning_content != None:
                reasoning_content += delta.reasoning_content
            else:
                # 开始回复
                if delta.content != "" and is_answering == False:
                    is_answering = True
                answer_content += delta.content
    return answer_content, reasoning_content


async def process_multiple_requests(questions, client, model, pbar):
    tasks = [call_client(question, client, model) for question in questions]

    async def wrapped_task(task):
        result = await task
        pbar.update(1)
        return result

    wrapped_tasks = [wrapped_task(task) for task in tasks]
    results = await asyncio.gather(*wrapped_tasks)

    return results

# def process_single_requests(questions, client, model):
#     task = call_client(questions[0], client, model)
#
#
#     return results

def process_requests(questions, client, model, async_num):
    pbar = tqdm(total=len(questions), desc='Processing')
    all_res = []
    for question in questions:
        res = process_single_request(question, client, model)
        pbar.update(1)
        all_res.append(res)
    pbar.close()
    return all_res


def async_process_requests(questions, client, model, async_num):
    pbar = tqdm(total=len(questions), desc='Processing')
    all_res = []
    for sub_questions in split_list(questions, async_num):
        res = asyncio.run(process_multiple_requests(sub_questions, client, model, pbar))
        all_res += res
    pbar.close()
    return all_res


# def process_requests(questions, client, model):

def split_list(lst, n):
    """
    使用生成器将长列表切分成长度为 n 的子列表
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# async def async_http_process_requests(requests, model_config, tasks=None, metadatas=None, save_file=None):
#     api_key = model_config['api_key']
#     base_url = model_config['base_url']
#     model_name = model_config['model_name']
#     num_concurrent = model_config['num_concurrent']
#     model = TemplateAPI(model=model_name, api_key=api_key, base_url=base_url, num_concurrent=num_concurrent)
#     sampling_params = {
#         'temperature': model_config['temperature'],
#     }
#
#     # 使用异步方法
#     outputs = await model.async_generate_until(
#         requests,
#         sampling_params,
#         tasks=tasks,
#         metadatas=metadatas,
#         save_file=save_file
#     )
#     return outputs
async def async_http_process_requests(requests, model_config, tasks=None, metadatas=None, save_file=None):
    api_key = model_config['api_key']
    base_url = model_config['base_url']
    model_name = model_config['model_name']
    num_concurrent = model_config['num_concurrent']

    # 使用异步 API 类
    model = AsyncTemplateAPI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        num_concurrent=num_concurrent
    )

    sampling_params = {
        'temperature': model_config['temperature'],
    }

    try:
        outputs = await model.async_generate_until(
            requests,
            sampling_params,
            tasks=tasks,
            metadatas=metadatas,
            save_file=save_file
        )
        return outputs
    except Exception as e:
        print(f"Error in async_http_process_requests: {e}")
        traceback.print_exc()
        return None

def async_vllm_process_requests(requests, model_config, tasks=None, metadatas=None, save_file=None):
    api_key = model_config['api_key']
    base_url = model_config['base_url']
    model_name = model_config['model_name']
    num_concurrent = model_config['num_concurrent']
    model = TemplateAPI(model=model_name, api_key=api_key, base_url=base_url, num_concurrent=num_concurrent)
    sampling_params = {
        'temperature': model_config['temperature'],
        'stop_token_ids': model_config['stop_token_ids'],
        'max_tokens': model_config['max_tokens'],
    }
    if 'Qwen3' in model_name:
        sampling_params["chat_template_kwargs"] = {"enable_thinking": False}

    outputs = model.generate_until(requests, sampling_params, tasks=tasks, metadatas=metadatas, save_file=save_file)
    return outputs


def simple_promptify(questions):
    requests = []
    for question in questions:
        requests.append([{'role': 'user', 'content': question}])
    return requests


def extract_between(input_string, before_string, after_string):
    pattern = rf'(?:{before_string}:|[*]{{0,2}}{before_string}:[*]{{0,2}}|[*]{{0,2}}{before_string}[*]{{0,2}}:?)' + \
        r'(.*?)' + \
        rf'(?:{after_string}:|[*]{{0,2}}{after_string}:[*]{{0,2}}|[*]{{0,2}}{after_string}[*]{{0,2}}:?)'
    match = re.search(pattern, input_string, re.DOTALL)
    if match:
        return match.group(1).strip()  # 提取匹配的内容并去掉多余空格
    else:
        return None  # 如果没有匹配到，返回 None


def extract_after(input_string, split_string):
    pattern = rf'(?:{split_string}:|[*]{{0,2}}{split_string}:[*]{{0,2}}|[*]{{0,2}}{split_string}[*]{{0,2}}:?)\s*(.*)'
    match = re.search(pattern, input_string, re.DOTALL)
    if match:
        return match.group(1).strip()  # 提取匹配的内容并去掉多余空格
    else:
        return None  # 如果没有匹配到，返回 None


def extract_before(input_string, split_string):
    pattern = rf'(.*)(?:{split_string}:|[*]{{0,2}}{split_string}:[*]{{0,2}}|[*]{{0,2}}{split_string}[*]{{0,2}}:?)'
    match = re.search(pattern, input_string, re.DOTALL)
    if match:
        return match.group(1).strip()  # 提取匹配的内容并去掉多余空格
    else:
        return None  # 如果没有匹配到，返回 None



def clean_string(input_string):
    input_string = input_string.strip().lower()
    input_string = re.sub(r' +', ' ', input_string)  # 多个空格到一个空格
    input_string = re.sub(r' \n', '\n', input_string).strip('"')  # 空格+换行 到 换行
    return input_string


def reduce_multi_empty_lines(input_string):
    return re.sub('\n+','\n', input_string)


def list_to_string(string_list, split_type):
    """
    将字符串列表组装成一个带编号的字符串。

    Args:
    string_list: 字符串列表。

    Returns:
    带编号的字符串。
    """
    numbered_list_str = ""
    for index, item in enumerate(string_list):
        if split_type == 'num':
            numbered_list_str += f"{index + 1}. {item}\n"
        else:
            numbered_list_str += f"{split_type} {item}\n"
    return numbered_list_str


def model_name_replacement(model_name):
    model_name = model_name.replace('gemma-2-9b-it@nvidia', 'gemma-2-9b-it')
    model_name = model_name.replace('gemma-2-9b-it@together', 'gemma-2-9b-it')
    model_name = model_name.replace('gemma-2-27b-it@together', 'gemma-2-27b-it')
    model_name = model_name.replace('gemma-2-27b-it@nvidia', 'gemma-2-27b-it')
    model_name = model_name.replace('deepseek-chat', 'deepseek-v2-chat-0628')
    model_name = model_name.replace('deepseek-coder', 'deepseek-v2-coder-0614')
    model_name = model_name.replace('DeepSeek-Coder-V2-0724', 'deepseek-v2-coder-0724')
    model_name = model_name.replace('Llama-3.1-405B-Inst-fp8', 'Llama-3.1-405B-Inst-fp8@together')
    model_name = model_name.replace('Llama-3.1-405B-Instruct-Turbo', 'Llama-3.1-405B-Inst-fp8@together')
    model_name = model_name.replace('Meta-Llama-3.1-405B-Instruct@hyperbolic', 'Llama-3.1-405B-Inst@hyperbolic')
    return model_name


def model_specific_extraction(model_name, prediction_str):
    if "Llama-3.1" in model_name:
        if "boxed" in prediction_str[-30:]:
            # print(prediction_str)
            # extract "$\boxed{36}$" --> 36
            # print(prediction_str)
            match = re.search(r'\\boxed{([\w\d]+)}', prediction_str)
            if match:
                return match.group(1)
    return None


def load_model_results(run_name_folders):
    model_results = {}

    for run_name, folder in run_name_folders.items():
        if not os.path.exists(folder):
            print(f"Folder {folder} does not exist.")
            continue
        # iterate all json files under the folder
        for filename in os.listdir(folder):
            filepath = os.path.join(folder, filename)
            if not filename.endswith(".json"):
                continue
            model_name = filename.replace(".json", "")
            model_name = f"{model_name}%{run_name}"
            model_results[model_name] = filepath
    return model_results


def extract_values_from_json(json_string, keys=["reasoning", "answer"], allow_no_quotes=False):
    extracted_values = {}
    for key in keys:
        # Create a regular expression pattern to find the value for the given key
        pattern = f'"{key}"\\s*:\\s*"([^"]*?)"'
        match = re.search(pattern, json_string)
        if match:
            extracted_values[key] = match.group(1)
        else:
            # Handle the case where the value might contain broken quotes
            pattern = f'"{key}"\\s*:\\s*"(.*?)"'
            match = re.search(pattern, json_string, re.DOTALL)
            if match:
                extracted_values[key] = match.group(1)
        if not match and allow_no_quotes:
            # to allow no quotes on the values
            pattern = f'"{key}"\\s*:\\s*([^,\\s]*)'
            match = re.search(pattern, json_string)
            if match:
                extracted_values[key] = match.group(1)
            else:
                # to allow no quotes on the keys
                pattern = f'{key}\\s*:\\s*([^,\\s]*)'
                match = re.search(pattern, json_string)
                if match:
                    extracted_values[key] = match.group(1)
    return extracted_values


def extract_first_complete_json(s):
    # Stack to keep track of opening and closing braces
    stack = []
    first_json_start = None

    for i, char in enumerate(s):
        if char == '{':
            stack.append(i)
            if first_json_start is None:
                first_json_start = i
        elif char == '}':
            if stack:
                start = stack.pop()
                if not stack:
                    # Complete JSON object found
                    first_json_str = s[first_json_start:i + 1]
                    try:
                        return json.loads(first_json_str.replace("\n", ""))
                    except json.JSONDecodeError:
                        return None
                    finally:
                        first_json_start = None
    return None


def extract_last_complete_json(s):
    # Stack to keep track of opening and closing braces
    stack = []
    last_json_start = None
    last_json_str = None

    for i, char in enumerate(s):
        if char == '{':
            stack.append(i)
            if last_json_start is None:
                last_json_start = i
        elif char == '}':
            if stack:
                start = stack.pop()
                if not stack:
                    # Complete JSON object found
                    last_json_str = s[last_json_start:i + 1]
                    last_json_start = None

    # Load the last JSON object
    if last_json_str:
        try:
            return json.loads(last_json_str.replace("\n", ""))
        except json.JSONDecodeError:
            pass

    return None
