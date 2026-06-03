import os
import base64
import json
import sys
import asyncio
import aiofiles
from tqdm import tqdm
from PIL import Image
import io
import shutil

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from config.paths import construct_dataset, random_dataset
from llm_api.model_config import get_configs, create_client
from llm_api.template_api import AsyncTemplateAPI

def get_base_prompt_template(index=0):
    templates = [
        """
        给定模因{{IMAGE}}，其内联文本为"{CUSTOM_TEXT}"，当其作为"{SCENE}"配图时，将它分类为有害或无害。
        评估标准:
        <中文有害模因是由图像和中文行内文字组成的多模态单元，具有对个人、组织、社区、社会群体或整个社会造成伤害的潜力。它们往往反映和强化了中国互联网上潜在的负面价值观和文化态度。>根据定义，符合以下任何一个标准的模因将被判定为有害。
        1. 模因表达了对特定个人或社会群体的厌恶、偏见或刻板印象。
        2. 模因包含了讽刺或粗鲁的内容，没有特定的目标。
        3. 模因暗示了性意图，以引起性冲动。
        4. 模因包含了颓废和绝望的情绪，或传达了一种自我否定的态度。
        评估步骤:
        1. 结合图像和文本分析模因的含义；
        2. 将模因与评估标准进行比较，检查是否有任何匹配；
        3. 只返回"有害"或"无害"的结论；
        """,
        """
        给定模因{{IMAGE}}，当其作为"{SCENE}"配图时，将它分类为有害或无害。
        评估标准:
        <中文有害模因是由图像和中文行内文字组成的多模态单元，具有对个人、组织、社区、社会群体或整个社会造成伤害的潜力。它们往往反映和强化了中国互联网上潜在的负面价值观和文化态度。>根据定义，符合以下任何一个标准的模因将被判定为有害。
        1. 模因表达了对特定个人或社会群体的厌恶、偏见或刻板印象。
        2. 模因包含了讽刺或粗鲁的内容，没有特定的目标。
        3. 模因暗示了性意图，以引起性冲动。
        4. 模因包含了颓废和绝望的情绪，或传达了一种自我否定的态度。
        评估步骤:
        1. 结合图像和文本分析模因的含义；
        2. 将模因与评估标准进行比较，检查是否有任何匹配；
        3. 只返回"有害"或"无害"的结论；
        """,
        """
        当一个模因作为"{SCENE}"配图时，将它分类为有害或无害。
        评估标准:
        <中文有害模因是由图像和中文行内文字组成的多模态单元，具有对个人、组织、社区、社会群体或整个社会造成伤害的潜力。它们往往反映和强化了中国互联网上潜在的负面价值观和文化态度。>根据定义，符合以下任何一个标准的模因将被判定为有害。
        1. 模因表达了对特定个人或社会群体的厌恶、偏见或刻板印象。
        2. 模因包含了讽刺或粗鲁的内容，没有特定的目标。
        3. 模因暗示了性意图，以引起性冲动。
        4. 模因包含了颓废和绝望的情绪，或传达了一种自我否定的态度。
        评估步骤:
        1. 结合图像和文本分析模因的含义；
        2. 将模因与评估标准进行比较，检查是否有任何匹配；
        3. 只返回"有害"或"无害"的结论；
        给定模因{{IMAGE}}，其内联文本为"{CUSTOM_TEXT}"，
        """,
        """
        给定模因{{IMAGE}}，其内联文本为"{CUSTOM_TEXT}"，将其分类为有害或无害。
        评估标准:
        <中文有害模因是由图像和中文行内文字组成的多模态单元，具有对个人、组织、社区、社会群体或整个社会造成伤害的潜力。它们往往反映和强化了中国互联网上潜在的负面价值观和文化态度。>根据定义，符合以下任何一个标准的模因将被判定为有害。
        1. 模因表达了对特定个人或社会群体的厌恶、偏见或刻板印象。
        2. 模因包含了讽刺或粗鲁的内容，没有特定的目标。
        3. 模因暗示了性意图，以引起性冲动。
        4. 模因包含了颓废和绝望的情绪，或传达了一种自我否定的态度。
        评估步骤:
        1. 结合图像和文本分析模因的含义；
        2. 将模因与评估标准进行比较，检查是否有任何匹配；
        3. 只返回"有害"或"无害"的结论；
        """,
        """
        给定模因{{IMAGE}}，将其分类为有害或无害。
        评估标准:
        <中文有害模因是由图像和中文行内文字组成的多模态单元，具有对个人、组织、社区、社会群体或整个社会造成伤害的潜力。它们往往反映和强化了中国互联网上潜在的负面价值观和文化态度。>根据定义，符合以下任何一个标准的模因将被判定为有害。
        1. 模因表达了对特定个人或社会群体的厌恶、偏见或刻板印象。
        2. 模因包含了讽刺或粗鲁的内容，没有特定的目标。
        3. 模因暗示了性意图，以引起性冲动。
        4. 模因包含了颓废和绝望的情绪，或传达了一种自我否定的态度。
        评估步骤:
        1. 结合图像和文本分析模因的含义；
        2. 将模因与评估标准进行比较，检查是否有任何匹配；
        3. 只返回"有害"或"无害"的结论；
        """,
        """
        给定模因，将它分类为有害或无害。
        评估标准:
        <中文有害模因是由图像和中文行内文字组成的多模态单元，具有对个人、组织、社区、社会群体或整个社会造成伤害的潜力。它们往往反映和强化了中国互联网上潜在的负面价值观和文化态度。>根据定义，符合以下任何一个标准的模因将被判定为有害。
        1. 模因表达了对特定个人或社会群体的厌恶、偏见或刻板印象。
        2. 模因包含了讽刺或粗鲁的内容，没有特定的目标。
        3. 模因暗示了性意图，以引起性冲动。
        4. 模因包含了颓废和绝望的情绪，或传达了一种自我否定的态度。
        评估步骤:
        1. 结合图像和文本分析模因的含义；
        2. 将模因与评估标准进行比较，检查是否有任何匹配；
        3. 只返回"有害"或"无害"的结论；
        给定模因{{IMAGE}}，其内联文本为"{CUSTOM_TEXT}"。
        """
    ]
    template = templates[index % len(templates)]
    return template


def create_full_prompt(meme_text, scene, feature="",index=0):
    """创建完整的提示词，包含社交媒体特征"""
    template = get_base_prompt_template(index=index)

    # 处理空值情况
    meme_text = meme_text if meme_text else ""
    scene = scene if scene else ""

    # 替换模板中的占位符
    full_prompt = template.replace("{CUSTOM_TEXT}", meme_text)
    full_prompt = full_prompt.replace("{SCENE}", scene)

    return full_prompt


async def load_json_data(json_path):
    """加载JSON文件数据"""
    try:
        async with aiofiles.open(json_path, 'r', encoding='utf-8') as f:
            content = await f.read()
            return json.loads(content)
    except FileNotFoundError:
        print(f" 未找到JSON文件: {json_path}")
        return None
    except Exception as e:
        print(f" 读取JSON失败 {json_path}: {e}")
        return None


async def encode_image_to_base64(image_path):
    """异步将图片转换为base64编码"""
    try:
        async with aiofiles.open(image_path, 'rb') as f:
            image_bytes = await f.read()

        with Image.open(io.BytesIO(image_bytes)) as img:
            img = img.convert('RGB')
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"图片编码失败 {image_path}: {e}")
        return None


async def process_single_json_with_thinking(json_path, meme_dir, client, storage_file, feature=""):
    """使用思考模式处理单个JSON文件"""
    # 加载JSON数据
    json_data = await load_json_data(json_path)
    if json_data is None:
        return None

    # 提取必要字段
    path_value = json_data.get('path', '')
    scene = json_data.get('scene', '')
    text = json_data.get('text', '')

    if not path_value:
        print(f" JSON文件缺少path字段: {json_path}")
        return None

    # 构造对应的meme图片路径
    meme_image_path = os.path.join(meme_dir, path_value)

    if not os.path.exists(meme_image_path):
        print(f" 未找到对应的meme图片: {meme_image_path}")
        return None

    # 编码图片
    base64_image = await encode_image_to_base64(meme_image_path)
    if base64_image is None:
        return None

    prompt_variants = []
    if storage_file == "with":
        prompt_variants = [
            lambda: create_full_prompt(text, scene, feature, 0),
            lambda: create_full_prompt(text, scene, feature, 1),
            lambda: create_full_prompt(text, scene, feature, 2)
        ]
    elif storage_file == "without":
        prompt_variants = [
            lambda: create_full_prompt(text, scene, feature, 3),
            lambda: create_full_prompt(text, scene, feature, 4),
            lambda: create_full_prompt(text, scene, feature, 5)
        ]

    # 准备meme_unset文件夹路径
    meme_unset_dir = os.path.join(os.path.dirname(meme_dir), 'meme_unset')
    os.makedirs(meme_unset_dir, exist_ok=True)

    # 尝试不同提示词
    for prompt_generator in prompt_variants:
        try:
            # 动态选择提示词变体
            prompt = prompt_generator()

            # 创建消息
            messages = await create_messages_with_embedded_image(prompt, base64_image)

            # 创建聊天完成
            completion = await client.chat.completions.create(
                model="qwen3-vl-plus",
                messages=messages,
                extra_body={"enable_thinking": True, "thinking_budget": 300},
                stream=True,
                stream_options={"include_usage": True},
                temperature=0.05,
                top_p=0.1
            )

            reasoning_content = ""
            answer_content = ""
            usage = None

            async for chunk in completion:
                if not chunk.choices:
                    if hasattr(chunk, 'usage') and chunk.usage:
                        usage = chunk.usage
                    continue

                delta = chunk.choices[0].delta

                if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
                    reasoning_content += delta.reasoning_content

                if hasattr(delta, "content") and delta.content:
                    answer_content += delta.content

            return {
                'content': answer_content,
                'reasoning': reasoning_content,
                'usage': usage.__dict__ if usage else {},
                'prompt_used': prompt,
                'original_data': json_data
            }

        except Exception as e:
            if "data_inspection_failed" in str(e):
                continue
            elif "rate_limit" in str(e):
                print("达到速率限制，等待重试")
                await asyncio.sleep(2)
                continue

    # 如果所有提示词都失败，将图片移动到meme_unset文件夹

    print(f"处理失败：{meme_image_path}")
    return None


async def save_result(result, json_filename, output_dir):
    """保存单个结果到JSON文件，使用原来的JSON文件名"""
    save_path = os.path.join(output_dir, json_filename)

    original_data = result.get('original_data', {})

    save_data = {
        'path': original_data.get('path', ''),
        'raw_label': original_data.get('raw_label', ''),
        'text': original_data.get('text', ''),
        'scene': original_data.get('scene', ''),
        'content': result.get('content', ''),
        'reason': result.get('reasoning', ''),
    }

    try:
        async with aiofiles.open(save_path, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(save_data, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"保存失败 {save_path}: {e}")


def get_user_range():
    """询问用户处理范围"""
    while True:
        try:
            choice = input("选择处理模式 (1: 全部处理, 2: 指定范围): ").strip()
            if choice == "1":
                return None, None  # 处理全部
            elif choice == "2":
                start = int(input("请输入起始序号: "))
                end = int(input("请输入结束序号: "))
                if start > end:
                    print("起始序号不能大于结束序号")
                    continue
                return start, end
            else:
                print("请输入1或2")
        except ValueError:
            print("请输入有效数字")


def find_json_files_in_range(json_dir, output_dir,storage_file,start=None, end=None):
    """遍历JSON目录，找到符合条件的JSON文件"""
    available_files = []

    if not os.path.exists(json_dir):
        print(f"JSON目录不存在: {json_dir}")
        return available_files

    # 获取已存在输出目录中的文件名集合
    existing_output_files = set(os.path.splitext(f)[0] for f in os.listdir(output_dir) if f.lower().endswith('.json')) if os.path.exists(output_dir) else set()

    # 遍历目录中的所有JSON文件
    for filename in os.listdir(json_dir):
        if filename.lower().endswith('.json'):
            json_path = os.path.join(json_dir, filename)
            base_name = os.path.splitext(filename)[0]

            if start is None or end is None:
                # 只添加输出目录中不存在的文件
                if base_name not in existing_output_files:
                    available_files.append((json_path, filename))
            else:
                # 尝试从文件名提取数字进行范围过滤
                try:
                    # 假设文件名格式为 "数字-P.json" 或 "数字-N.json"
                    if '-' in base_name:
                        number_part = base_name.split('-')[0]
                        # 处理可能的括号，如 "123(1)-P"
                        if '(' in number_part:
                            number_part = number_part.split('(')[0]
                        file_number = int(number_part)

                        if start <= file_number <= end and base_name not in existing_output_files:
                            available_files.append((json_path, filename))
                    else:
                        if '.' in base_name:
                            number_part = base_name.split('.')[0]
                            file_number = int(number_part)
                            if start <= file_number <= end and base_name not in existing_output_files:
                                available_files.append((json_path, filename))
                except ValueError:
                    # 如果无法提取数字，跳过该文件
                    continue

    # 按文件名排序
    available_files.sort(key=lambda x: x[1])

    if storage_file == "without":
        # 根据数字部分去重
        unique_files = {}
        for json_path, filename in available_files:
            # 提取数字部分
            try:
                if '-' in filename:
                    number_part = filename.split('-')[0]
                    if '(' in number_part:
                        number_part = number_part.split('(')[0]
                else:
                    number_part = filename.split('.')[0]

                # 只保留第一个出现的文件（-P或-N）
                if number_part not in unique_files:
                    unique_files[number_part] = (json_path, filename)
            except:
                continue

        # 更新available_files为去重后的结果
        available_files = list(unique_files.values())

    return available_files


async def create_messages_with_embedded_image(prompt, base64_image):
    """创建包含嵌入图片的消息，支持图片占位符"""

    # 检查是否包含图片占位符
    if "{{IMAGE}}" in prompt:
        # 分离文本，图片嵌入中间
        parts = prompt.split("{{IMAGE}}")

        content = []

        # 添加前半部分文本
        if parts[0].strip():
            content.append({
                "type": "text",
                "text": parts[0].strip()
            })

        # 添加图片
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        })

        # 添加后半部分文本
        if len(parts) > 1 and parts[1].strip():
            content.append({
                "type": "text",
                "text": parts[1].strip()
            })

    else:
        # 传统方式：文本在前，图片在后
        content = [
            {
                "type": "text",
                "text": prompt
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
        ]

    return [{
        "role": "user",
        "content": content
    }]


async def main(storage_file):
    """主函数"""
    print(" Qwen3-VL-Plus ")
    print("=" * 70)

    json_dir = str(construct_dataset("restructured"))
    meme_dir = str(construct_dataset("meme"))
    feature_file = str(construct_dataset("feature.txt"))

    # 检查目录是否存在
    if not os.path.exists(json_dir):
        print(f" JSON目录不存在: {json_dir}")
        return

    if not os.path.exists(meme_dir):
        print(f" Meme目录不存在: {meme_dir}")
        return

    feature_text = ""
    if os.path.exists(feature_file):
        try:
            async with aiofiles.open(feature_file, 'r', encoding='utf-8') as f:
                feature_text = await f.read()
        except Exception as e:
            print(f" 读取特征文件失败: {e}")
            print(" 将使用空特征继续处理")
    else:
        print(f" 特征文件不存在: {feature_file}")
        print(" 将使用空特征继续处理")

    output_dir=""
    # 创建输出目录
    if storage_file == 'with':
        output_dir = str(construct_dataset("LLM_with_content"))
    elif storage_file == 'without':
        output_dir = str(construct_dataset("LLM_without_content"))
    os.makedirs(output_dir, exist_ok=True)

    # 获取用户配置
    start, end = get_user_range()

    print(f"\n 配置信息:")
    if start is None:
        print("- 处理模式: 全部文件")
    else:
        print(f"- 处理范围: {start} - {end}")
    print(f"- JSON目录: {json_dir}")
    print(f"- Meme目录: {meme_dir}")
    print(f"- 输出目录: {output_dir}")

    # 遍历JSON目录，找到符合条件的文件
    available_files = find_json_files_in_range(json_dir,output_dir,storage_file,start, end)

    if not available_files:
        if start is None:
            print(" 没有找到任何JSON文件")
        else:
            print(f" 没有找到符合范围 {start}-{end} 的JSON文件")
        return

    if start is None:
        print(f" 找到 {len(available_files)} 个JSON文件")
    else:
        print(f" 找到 {len(available_files)} 个JSON文件在范围 {start}-{end} 内")

    # 检查对应的meme图片文件
    missing_meme_files = []
    valid_files = []

    for json_path, filename in available_files:
        json_data = await load_json_data(json_path)
        if json_data and 'path' in json_data:
            meme_path = os.path.join(meme_dir, json_data['path'])
            if os.path.exists(meme_path):
                valid_files.append((json_path, filename))
            else:
                missing_meme_files.append(json_data['path'])

    if missing_meme_files:
        print(f" 缺少meme图片文件: {len(missing_meme_files)}个")
        print("将跳过这些文件的处理")

    if not valid_files:
        print(" 没有找到可处理的有效文件")
        return

    print(f" 实际可处理文件: {len(valid_files)} 个")

    # 初始化客户端
    client_config = get_configs('aliyun')
    client = create_client(client_config)

    # 并发控制
    semaphore = asyncio.Semaphore(50)

    async def process_with_semaphore(json_path, filename):
        """带并发控制的处理函数"""
        async with semaphore:
            result = await process_single_json_with_thinking(json_path, meme_dir, client,storage_file,feature_text)

            if result:
                await save_result(result, filename, output_dir)
                return filename
            return None

    # 创建进度条
    pbar = tqdm(total=len(valid_files), desc='处理文件', unit='个')

    # 并发处理所有文件
    tasks = []
    for json_path, filename in valid_files:
        task = asyncio.create_task(process_with_semaphore(json_path, filename))
        task.add_done_callback(lambda _: pbar.update(1))
        tasks.append(task)

    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        pbar.close()

        # 统计结果
        success_count = sum(1 for r in results if r and not isinstance(r, Exception))
        error_count = len(results) - success_count

        print(f"\n 处理完成:")
        print(f"- 总文件数: {len(valid_files)}")
        print(f"- 成功处理: {success_count}")
        print(f"- 处理失败: {error_count}")
        if missing_meme_files:
            print(f"- 跳过文件: {len(missing_meme_files)} (缺少对应meme图片)")
        print(f"- 结果保存在: {output_dir}")

    except KeyboardInterrupt:
        print("\n 用户中断处理")
        pbar.close()
    except Exception as e:
        print(f"\n 处理过程出错: {e}")
        pbar.close()


if __name__ == "__main__":
    asyncio.run(main("with"))