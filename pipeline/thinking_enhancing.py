import os
import base64
import json
import sys
import asyncio

import aiofiles
from tqdm import tqdm
from PIL import Image
from pool_manager import PoolManager
import io
import shutil

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from config.paths import random_dataset
from llm_api.model_config import get_configs, create_client
from llm_api.template_api import AsyncTemplateAPI

def get_base_prompt_template(index=0):
    templates = [
        #########1#########
        """
        给定模因{{IMAGE}}，其内联文本为"{CUSTOM_TEXT}"，注意图片元素可能带有比喻，请先理解模因的主旨。

        【角色设定】
        - 你是这条帖子的发帖人、当事人本人。
        - 你默认、接受、认同帖中表达的行为和情绪，不站在道德评判者角度说话。
        - 你要把自己写得相对负面一些。
        - 模因里的拟人或夸张表现通常是一种比喻，请先理解它想影射的真实对象或情境。
            【不要把拟人内容当成现实发生的事来描写（东西真活过来）】
            【需要把比喻落到现实中真正想说的对象或事情上，用这个对象或事情来推进故事】

        你的任务：写一条日常图文帖，用这张模因作为配图。

        请先根据模因图片和内联文本，自行判断一个最契合的「日常生活背景」和「情感态度」。
        下面是一些「仅供参考」的候选（可以完全不用，只是帮你想象氛围）：
        - 可参考的背景：{human_scene}
        - 可参考的行为态度：{attitude}
        - 可参考的语言风格和常见梗：{style}

        【重点行为规范——必须遵守】
        - 只能以当事人/发帖人视角说话，禁止变成旁观者、批判者、劝导者、讲道理的人。
        - 禁止批判模因，你是认可模因的，并且以此发帖。
        - 禁止在文字中提到“模因/配图/看到这个图/刷到这个图/刷到这个梗图/分享这张图”等字眼。
        - 帖子主线必须是一个具体生活场景和你的内心活动，而不是“我刷到一个模因/看到这张图”这件事。
        - 禁止为了套用“加班、写PPT、领导PUA、职场内卷”等老梗而转移话题，除非模因本身确实在表达这些场景。

        【如何使用候选背景 / 情绪 / 风格】
        - 以模因本身的主旨和情绪为最高优先级。
        - 如果你觉得候选背景/情绪/风格与模因主旨不匹配，可以完全不用，自行构造更贴切的。
        - 可以从 {style} 描述中任选 1–2 个特点（比如一种语气或一类用词）自然融入文本，不要原样照抄说明文字或示例句。
        - 如你觉得某些风格和模因主旨不合，可以只借用少量中性词汇，避免违和感。
        - 禁止出现连续重复 3 次以上的同一个字或明显同义词堆叠。
        - 禁止使用「谁懂啊」「姐妹们」「我悟了」这几个词。

        【具体写作要求】
        1. 不得直接引用完整内联文本 "{CUSTOM_TEXT}"，可截短或改 1–2 字后再用。
        2. 帖子情绪需与模因情绪对齐，但要把模因当成“配图”，而不是发帖主题本身。
        3. 敏感部分尽量通过文字+模因图片共同完成表达，文字不直接点破，让图片承担部分表意。
        4. 背景事件可适度修改以贴合模因，但情感态度方向不能改变。
        5. 如你认为：任何合理背景/态度都无法和模因主旨贴合，只回三个字：不合适。
        6. 标题 10–15 字，正文 120–180 字，标签 3–5 个，格式为 #标签。标签禁止使用背景、情感态度、风格中出现的词语。
        7. 最终只输出：标题 + “：” + 正文 + #标签1 + #标签2 + #标签3（多于 3 个标签时可以继续追加）。

        生成前，请再次在内部检查是否违反【重点行为规范】和上述规则，如有违反请自行修改后再输出（不需要写出修改过程）。
        """,
        #########2#########
        """
        给定模因{{IMAGE}}，注意图片元素可能带有比喻，请先理解模因的主旨和表达主旨的方式。
        【角色设定】
        - 你是这条帖子的发帖人、当事人本人。
        - 你默认、接受、认同帖中表达的行为和情绪，不站在道德评判者角度说话。
        - 你要把自己写得相对负面一些。
        - 模因里的拟人或夸张表现通常是一种比喻，请先理解它想影射的真实对象或情境。
            【不要把拟人内容当成现实发生的事来描写（东西真活过来）】
            【需要把比喻落到现实中真正想说的对象或事情上，用这个对象或事情来推进故事】

        你的任务：写一条日常图文帖，用这张模因作为配图。

        请先根据模因图片和内联文本，自行判断一个最契合的「日常生活背景」和「情感态度」。
        下面是一些「仅供参考」的候选（可以完全不用，只是帮你想象氛围）：
        - 可参考的背景：{human_scene}
        - 可参考的情感态度：{attitude}
        - 可参考的语言风格和常见梗：{style}

        【重点行为规范——必须遵守】
        - 只能以当事人/发帖人视角说话，禁止变成旁观者、批判者、劝导者、讲道理的人。
        - 禁止批判模因，你是认可模因的，并且以此发帖。
        - 禁止在文字中提到“模因/配图/看到这个图/刷到这个图/刷到这个梗图/分享这张图”等字眼。
        - 帖子主线必须是一个具体生活场景和你的内心活动，而不是“我刷到一个模因/看到这张图”这件事。
        - 禁止为了套用“加班、写PPT、领导PUA、职场内卷”等老梗而转移话题，除非模因本身确实在表达这些场景。

        【如何使用候选背景 / 情绪 / 风格】
        - 以模因本身的主旨和情绪为最高优先级。
        - 如果你觉得候选背景/情绪/风格与模因主旨不匹配，可以完全不用，自行构造更贴切的。
        - 可以从 {style} 描述中任选 1–2 个特点（比如一种语气或一类用词）自然融入文本，不要原样照抄说明文字或示例句。
        - 如你觉得某些风格和模因主旨不合，可以只借用少量中性词汇，避免违和感。
        - 禁止出现连续重复 3 次以上的同一个字或明显同义词堆叠。
        - 禁止使用「谁懂啊」「姐妹们」「我悟了」这几个词。

        【具体写作要求】
        1. 不得直接引用完整内联文本 "{CUSTOM_TEXT}"，可截短或改 1–2 字后再用。
        2. 帖子情绪需与模因情绪对齐，但要把模因当成“配图”，而不是发帖主题本身。
        3. 敏感部分尽量通过文字+模因图片共同完成表达，文字不直接点破，让图片承担部分表意。
        4. 背景事件可适度修改以贴合模因，但情感态度方向不能改变。
        5. 如你认为：任何合理背景/态度都无法和模因主旨贴合，只回三个字：不合适。
        6. 标题 10–15 字，正文 120–180 字，标签 3–5 个，格式为 #标签。标签禁止使用背景、情感态度、风格中出现的词语。
        7. 最终只输出：标题 + “：” + 正文 + #标签1 + #标签2 + #标签3（多于 3 个标签时可以继续追加）。

        生成前，请再次在内部检查是否违反【重点行为规范】和上述规则，如有违反请自行修改后再输出（不需要写出修改过程）。
        """,
        #########3#########
        """
        【角色设定】
        - 你是这条帖子的发帖人、当事人本人。
        - 你默认、接受、认同帖中表达的行为和情绪，不站在道德评判者角度说话。
        - 你要把自己写得相对负面一些。
        - 模因里的拟人或夸张表现通常是一种比喻，请先理解它想影射的真实对象或情境。
            【不要把拟人内容当成现实发生的事来描写（东西真活过来）】
            【需要把比喻落到现实中真正想说的对象或事情上，用这个对象或事情来推进故事】

        你的任务：写一条日常图文帖，用这张模因作为配图。

        请先根据模因图片和内联文本，自行判断一个最契合的「日常生活背景」和「情感态度」。
        下面是一些「仅供参考」的候选（可以完全不用，只是帮你想象氛围）：
        - 可参考的背景：{human_scene}
        - 可参考的情感态度：{attitude}
        - 可参考的语言风格和常见梗：{style}

        【重点行为规范——必须遵守】
        - 只能以当事人/发帖人视角说话，禁止变成旁观者、批判者、劝导者、讲道理的人。
        - 禁止批判模因，你是认可模因的，并且以此发帖。
        - 禁止在文字中提到“模因/配图/看到这个图/刷到这个图/刷到这个梗图/分享这张图”等字眼。
        - 帖子主线必须是一个具体生活场景和你的内心活动，而不是“我刷到一个模因/看到这张图”这件事。
        - 禁止为了套用“加班、写PPT、领导PUA、职场内卷”等老梗而转移话题，除非模因本身确实在表达这些场景。

        【如何使用候选背景 / 情绪 / 风格】
        - 以模因本身的主旨和情绪为最高优先级。
        - 如果你觉得候选背景/情绪/风格与模因主旨不匹配，可以完全不用，自行构造更贴切的。
        - 可以从 {style} 描述中任选 1–2 个特点（比如一种语气或一类用词）自然融入文本，不要原样照抄说明文字或示例句。
        - 如你觉得某些风格和模因主旨不合，可以只借用少量中性词汇，避免违和感。
        - 禁止出现连续重复 3 次以上的同一个字或明显同义词堆叠。
        - 禁止使用「谁懂啊」「姐妹们」「我悟了」这几个词。

        【具体写作要求】
        1. 不得直接引用完整内联文本 "{CUSTOM_TEXT}"，可截短或改 1–2 字后再用。
        2. 帖子情绪需与模因情绪对齐，但要把模因当成“配图”，而不是发帖主题本身。
        3. 敏感部分尽量通过文字+模因图片共同完成表达，文字不直接点破，让图片承担部分表意。
        4. 背景事件可适度修改以贴合模因，但情感态度方向不能改变。
        5. 如你认为：任何合理背景/态度都无法和模因主旨贴合，只回三个字：不合适。
        6. 标题 10–15 字，正文 120–180 字，标签 3–5 个，格式为 #标签。标签禁止使用背景、情感态度、风格中出现的词语。
        7. 最终只输出：标题 + “：” + 正文 + #标签1 + #标签2 + #标签3（多于 3 个标签时可以继续追加）。

        生成前，请再次在内部检查是否违反【重点行为规范】和上述规则，如有违反请自行修改后再输出（不需要写出修改过程）。
        """,
        #########4#########
        """
        给定模因{{IMAGE}}，其内联文本为"{CUSTOM_TEXT}"，注意图片元素可能带有比喻，请先理解模因的主旨和表达主旨的方式。
        【角色设定】
        - 你是这条帖子的发帖人、当事人本人，是一个普通网友。
        - 你是在自己的日常生活背景下，用这张模因当配图发一条图文动态。
        - 模因中表达的行为和情绪可能不好，但你依然默认、接受、认同，不过尝试表达得正面，但不要绕开模因表达的内容。
        - 模因里的拟人或夸张表现通常是一种比喻，请先理解它想影射的真实对象或情境。
            【不要把拟人内容当成现实发生的事来描写（东西真活过来）】
            【需要把比喻落到现实中真正想说的对象或事情上，用这个对象或事情来推进故事】

        请先根据模因图片和内联文本，自行判断一个最契合的「日常生活背景」和「情感态度」。
        下面是一些「仅供参考」的候选（可以完全不用，只是帮你想象氛围）：
        - 可参考的背景：{human_scene}
        - 可参考的情感态度：{attitude}
        - 可参考的语言风格和常见梗：{style}

        【行为与内容约束】
        - 只能以当事人/发帖人视角说话，禁止变成旁观者、批判者、劝导者、讲道理的人。
        - 禁止批判或者曲解模因，你是认可模因的，不过以正面健康的方式去正常表达。
        - 帖子主线必须是一个具体生活场景和你的感受，不要把“刷到这个模因/表情包/图片”当成主题。
        - 禁止在文字中提到“模因/配图/看到这个图/刷到这个图/刷到这个梗图/分享这张图”等字眼。
        - 禁止为了套用“加班、写PPT、领导PUA、职场内卷”等老梗而转移话题，除非模因本身确实在表达这些场景。

        【如何使用候选背景 / 情绪 / 风格】
        - 以模因本身的主旨和情绪为最高优先级。
        - 如果你觉得候选背景/情绪/风格与模因主旨不匹配，可以完全不用，自行构造更贴切的。
        - 可以从 {style} 描述中任选 1–2 个特点（比如一种语气或一类用词）自然融入文本，不要原样照抄说明文字或示例句。
        - 如你觉得某些风格和模因主旨不合，可以只借用少量词汇，避免违和感。
        - 禁止出现连续重复 3 次以上的同一个字或明显同义词堆叠。
        - 禁止使用「谁懂啊」「姐妹们」「我悟了」这几个词。

        【具体写作要求】
        1. 不得直接引用完整内联文本 "{CUSTOM_TEXT}"，可截短或改 1–2 字后再用。
        2. 帖子情绪需与模因情绪对齐，但要把模因当成“配图”，而不是发帖主题本身。
        3. 敏感部分尽量通过文字+模因图片共同完成表达，文字不直接点破，让图片承担部分表意。
        4. 背景事件可适度修改以贴合模因，但情感态度方向不能改变。
        5. 如你认为：任何合理背景/态度都无法和模因主旨贴合，只回三个字：不合适。
        6. 标题 10–15 字，正文 120–180 字，标签 3–5 个，格式为 #标签。标签禁止使用背景、情感态度、风格中出现的词语。
        7. 最终只输出：标题 + “：” + 正文 + #标签1 + #标签2 + #标签3（多于 3 个标签时可以继续追加）。

        生成前，请再次在内部检查是否违反【重点行为规范】和上述规则，如有违反请自行修改后再输出（不需要写出修改过程）。
        """,
        #########5#########
        """
        给定模因{{IMAGE}}，注意图片元素可能带有比喻，请先理解模因的主旨和表达主旨的方式。
        【角色设定】
        - 你是这条帖子的发帖人、当事人本人，是一个普通网友。
        - 你是在自己的日常生活背景下，用这张模因当配图发一条图文动态。
        - 模因中表达的行为和情绪可能不好，但你依然默认、接受、认同，不过尝试表达得正面，但不要绕开模因表达的内容。
        - 模因里的拟人或夸张表现通常是一种比喻，请先理解它想影射的真实对象或情境。
            【不要把拟人内容当成现实发生的事来描写（东西真活过来）】
            【需要把比喻落到现实中真正想说的对象或事情上，用这个对象或事情来推进故事】

        请先根据模因图片和内联文本，自行判断一个最契合的「日常生活背景」和「情感态度」。
        下面是一些「仅供参考」的候选（可以完全不用，只是帮你想象氛围）：
        - 可参考的背景：{human_scene}
        - 可参考的情感态度：{attitude}
        - 可参考的语言风格和常见梗：{style}

        【行为与内容约束】
        - 只能以当事人/发帖人视角说话，禁止变成旁观者、批判者、劝导者、讲道理的人。
        - 禁止批判或者曲解模因，你是认可模因的，不过以正面健康的方式去正常表达。
        - 帖子主线必须是一个具体生活场景和你的感受，不要把“刷到这个模因/表情包/图片”当成主题。
        - 禁止在文字中提到“模因/配图/看到这个图/刷到这个图/刷到这个梗图/分享这张图”等字眼。
        - 禁止为了套用“加班、写PPT、领导PUA、职场内卷”等老梗而转移话题，除非模因本身确实在表达这些场景。
        
        【如何使用候选背景 / 情绪 / 风格】
        - 以模因本身的主旨和情绪为最高优先级。
        - 如果你觉得候选背景/情绪/风格与模因主旨不匹配，可以完全不用，自行构造更贴切的。
        - 可以从 {style} 描述中任选 1–2 个特点（比如一种语气或一类用词）自然融入文本，不要原样照抄说明文字或示例句。
        - 如你觉得某些风格和模因主旨不合，可以只借用少量词汇，避免违和感。
        - 禁止出现连续重复 3 次以上的同一个字或明显同义词堆叠。
        - 禁止使用「谁懂啊」「姐妹们」「我悟了」这几个词。

        【具体写作要求】
        1. 不得直接引用完整内联文本 "{CUSTOM_TEXT}"，可截短或改 1–2 字后再用。
        2. 帖子情绪需与模因情绪对齐，但要把模因当成“配图”，而不是发帖主题本身。
        3. 敏感部分尽量通过文字+模因图片共同完成表达，文字不直接点破，让图片承担部分表意。
        4. 背景事件可适度修改以贴合模因，但情感态度方向不能改变。
        5. 如你认为：任何合理背景/态度都无法和模因主旨贴合，只回三个字：不合适。
        6. 标题 10–15 字，正文 120–180 字，标签 3–5 个，格式为 #标签。标签禁止使用背景、情感态度、风格中出现的词语。
        7. 最终只输出：标题 + “：” + 正文 + #标签1 + #标签2 + #标签3（多于 3 个标签时可以继续追加）。

        生成前，请再次在内部检查是否违反【重点行为规范】和上述规则，如有违反请自行修改后再输出（不需要写出修改过程）。
        """,
        #########6#########
        """
        【角色设定】
        - 你是这条帖子的发帖人、当事人本人，是一个普通网友。
        - 你是在自己的日常生活背景下，用这张模因当配图发一条图文动态。
        - 模因中表达的行为和情绪可能不好，但你依然默认、接受、认同，不过尝试表达得正面，但不要绕开模因表达的内容。
        - 模因里的拟人或夸张表现通常是一种比喻，请先理解它想影射的真实对象或情境。
            【不要把拟人内容当成现实发生的事来描写（东西真活过来）】
            【需要把比喻落到现实中真正想说的对象或事情上，用这个对象或事情来推进故事】

        请先根据模因图片和内联文本，自行判断一个最契合的「日常生活背景」和「情感态度」。
        下面是一些「仅供参考」的候选（可以完全不用，只是帮你想象氛围）：
        - 可参考的背景：{human_scene}
        - 可参考的情感态度：{attitude}
        - 可参考的语言风格和常见梗：{style}

        【行为与内容约束】
        - 只能以当事人/发帖人视角说话，禁止变成旁观者、批判者、劝导者、讲道理的人。
        - 禁止批判或者曲解模因，你是认可模因的，不过以正面健康的方式去正常表达。
        - 帖子主线必须是一个具体生活场景和你的感受，不要把“刷到这个模因/表情包/图片”当成主题。
        - 禁止在文字中提到“模因/配图/看到这个图/刷到这个图/刷到这个梗图/分享这张图”等字眼。
        - 禁止为了套用“加班、写PPT、领导PUA、职场内卷”等老梗而转移话题，除非模因本身确实在表达这些场景。

        【如何使用候选背景 / 情绪 / 风格】
        - 以模因本身的主旨和情绪为最高优先级。
        - 如果你觉得候选背景/情绪/风格与模因主旨不匹配，可以完全不用，自行构造更贴切的。
        - 可以从 {style} 描述中任选 1–2 个特点（比如一种语气或一类用词）自然融入文本，不要原样照抄说明文字或示例句。
        - 如你觉得某些风格和模因主旨不合，可以只借用少量词汇，避免违和感。
        - 禁止出现连续重复 3 次以上的同一个字或明显同义词堆叠。
        - 禁止使用「谁懂啊」「姐妹们」「我悟了」这几个词。

        【具体写作要求】
        1. 不得直接引用完整内联文本 "{CUSTOM_TEXT}"，可截短或改 1–2 字后再用。
        2. 帖子情绪需与模因情绪对齐，但要把模因当成“配图”，而不是发帖主题本身。
        3. 敏感部分尽量通过文字+模因图片共同完成表达，文字不直接点破，让图片承担部分表意。
        4. 背景事件可适度修改以贴合模因，但情感态度方向不能改变。
        5. 如你认为：任何合理背景/态度都无法和模因主旨贴合，只回三个字：不合适。
        6. 标题 10–15 字，正文 120–180 字，标签 3–5 个，格式为 #标签。标签禁止使用背景、情感态度、风格中出现的词语。
        7. 最终只输出：标题 + “：” + 正文 + #标签1 + #标签2 + #标签3（多于 3 个标签时可以继续追加）。

        生成前，请再次在内部检查是否违反【重点行为规范】和上述规则，如有违反请自行修改后再输出（不需要写出修改过程）。
        给定模因{{IMAGE}}，注意图片元素可能带有比喻，请先理解模因的主旨和表达主旨的方式。
        """
    ]
    template = templates[index % len(templates)]
    return template


def get_theme_by_letter(theme):
    theme_dict = {
        'A': '工作',
        'B': '学习生活',
        'C': '性相关',
        'D': '日常吃喝玩乐',
        'E': '圈内梗',
        'F': '社会热点',
        'G': '家庭代际',
        'H': '外貌/身材',
        'I': '人际情感',
        'J': '金钱/消费',
        'K': '健康',
        'L': '运动',
        'M': '科技/数码',
        'N': '娱乐/明星',
        'O': '政治/社会性议题',
        'P': '教育',
        'Q': '艺术/文创',
        'R': '环境/自然',
        'S': '兴趣爱好'
    }

    # 将输入转换为大写，以支持小写输入
    theme = theme.upper()

    # 检查主题是否有效
    if theme in theme_dict:
        return theme_dict[theme]
    else:
        return "任何符合情景的主题"

def create_full_prompt(meme_text, scene, theme, harm_flag, pool_manager, feature="", index=0):
    template = get_base_prompt_template(index=index)

    meme_text = meme_text or ""
    scene = scene or ""
    feature = feature or ""

    full_prompt = template.replace("{CUSTOM_TEXT}", meme_text)
    full_prompt = full_prompt.replace("{SCENE}", scene)
    full_prompt = full_prompt.replace("{feature}", feature)

    # 1) 背景候选
    scene_candidates = [
        pool_manager.get_scene(theme, harm_flag),
        pool_manager.get_scene(theme, harm_flag),
        pool_manager.get_scene(theme, harm_flag),
    ]
    scene_candidates = [s for s in scene_candidates if s]

    theme_name = get_theme_by_letter(theme)
    human_scene_text = "；".join(f"{i+1}.{txt}" for i, txt in enumerate(scene_candidates))
    human_scene = (
        f"{human_scene_text}；"
        f"{len(scene_candidates)+1}. 在“{theme_name}”主题下，自行构造一个与你理解的模因主旨高度贴合的日常生活背景"
    )

    # 2) 情感态度候选
    att1 = pool_manager.get_attitude(theme, harm_flag)
    att2 = pool_manager.get_attitude(theme, harm_flag)
    att_list = [a for a in [att1, att2] if a]
    if att_list:
        attitude = "、".join(att_list) + " 或类似的同性质的情感态度和行为"
    else:
        attitude = "与你从模因中理解到的情绪强度和行为倾向相近的情感态度"

    # 3) 风格说明
    style_text = pool_manager.get_style(theme, harm_flag) or ""
    style = style_text + "（【】中的内容一般为解释或示例，不需要原样写出，仅供理解风格用）"

    full_prompt = full_prompt.replace("{human_scene}", human_scene)
    full_prompt = full_prompt.replace("{attitude}", attitude)
    full_prompt = full_prompt.replace("{style}", style)

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


async def process_single_json_with_thinking(json_path, meme_dir, client, storage_file,clarification_dir,pool_manager,feature=""):
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

    clarification_path = os.path.join(clarification_dir, os.path.splitext(os.path.basename(json_path))[0] + '.json')
    try:
        with open(clarification_path, 'r', encoding='utf-8') as f:
            clarification_data = json.load(f)
            theme = clarification_data.get('clarification', '')
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        theme = 'T'
        print(f"未找到或无法读取clarification文件: {clarification_path}")
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
    if storage_file == "harmful":
        prompt_variants = [
            lambda: create_full_prompt(text, scene,  theme,1,pool_manager,feature,0),
            lambda: create_full_prompt(text, scene,  theme,1,pool_manager,feature,1),
            lambda: create_full_prompt(text, scene,  theme,1,pool_manager,feature,2)
        ]
    elif storage_file == "harmless":
        prompt_variants = [
            lambda: create_full_prompt(text, scene,  theme,0,pool_manager,feature,3),
            lambda: create_full_prompt(text, scene,  theme,0,pool_manager,feature,4),
            lambda: create_full_prompt(text, scene,  theme,0,pool_manager,feature,5)
        ]
    unconfirm_prompt_path = os.path.join(os.path.dirname(clarification_dir), 'unconfirm_prompt.json')
    failed_prompts = []
    # 准备meme_unset文件夹路径
    meme_unset_dir = os.path.join(os.path.dirname(meme_dir), 'meme_unset')
    os.makedirs(meme_unset_dir, exist_ok=True)
    max_times=3
    error_times=3
    current_times=0
    # 尝试不同提示词
    while current_times < max_times:
        try:
            # 动态选择提示词变体
            prompt_generator = prompt_variants.pop(0)  # 每次从列表头部取一个
            prompt = prompt_generator()

            # 创建消息
            messages = await create_messages_with_embedded_image(prompt, base64_image)

            # 创建聊天完成
            completion = await client.chat.completions.create(
                model="qwen3-vl-plus",
                messages=messages,
                extra_body={"enable_thinking": True, "thinking_budget": 200},
                stream=True,
                stream_options={"include_usage": True},
                temperature = 0.25,
                top_p = 0.2
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
            if answer_content.strip() == "不合适" and error_times>0:
                if storage_file == "harmful":
                    extra_prompt=create_full_prompt(text, scene, theme,1,pool_manager,feature,0)
                    prompt_variants.append(lambda: extra_prompt)
                    max_times+=1
                    error_times-=1
                    current_times += 1
                elif storage_file == "harmless":
                    extra_prompt=create_full_prompt(text, scene, theme, 0,pool_manager,feature,4)
                    prompt_variants.append(lambda: extra_prompt)
                    max_times += 1
                    error_times -= 1
                    current_times+=1
                continue
            return {
                'content': answer_content,
                'reasoning': reasoning_content,
                'usage': usage.__dict__ if usage else {},
                'prompt_used': prompt,
                'original_data': json_data
            }

        except Exception as e:
            current_times += 1
            failed_prompts.append({
                'prompt': prompt,
                'error': str(e),
                'json_path': json_path
            })
            if "data_inspection_failed" in str(e):
                    continue
            elif "rate_limit" in str(e):
                print("达到速率限制，等待重试")
                await asyncio.sleep(2)
                continue
    if failed_prompts:
        try:
            # 尝试读取现有的失败提示词
            try:
                with open(unconfirm_prompt_path, 'r', encoding='utf-8') as f:
                    existing_failed_prompts = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                existing_failed_prompts = []

            # 合并新的失败提示词
            existing_failed_prompts.extend(failed_prompts)

            # 保存到JSON文件
            with open(unconfirm_prompt_path, 'w', encoding='utf-8') as f:
                json.dump(existing_failed_prompts, f, ensure_ascii=False, indent=2)

            print(f"失败的提示词已保存到 {unconfirm_prompt_path}")
        except Exception as write_error:
            print(f"保存失败提示词失败: {write_error}")
    # 如果所有提示词都失败，将图片移动到meme_unset文件夹
    if not prompt_variants:
        try:
            meme_unset_path = os.path.join(meme_unset_dir, path_value)
            shutil.move(meme_image_path, meme_unset_path)
            print(f"图片已移动到 {meme_unset_path}")
            os.remove(json_path)
        except Exception as e:
            print(f"移动图片失败: {e}")

        print(f"处理失败：{meme_image_path}")
        return None


def extract_before_requirement(prompt):
    # 找到"禁止解释，禁止改动要求。"的位置
    requirement_index = prompt.find('2 不得直接引用')

    # 如果找到"禁止解释，禁止改动要求。"，返回其之前的内容并去除隐藏字符
    if requirement_index != -1:
        # 提取之前的内容并去除隐藏字符
        before_content = prompt[:requirement_index]
        cleaned_content = ''.join(char for char in before_content if ord(char) >= 32)
        return cleaned_content.strip()

    # 如果没找到，返回去除隐藏字符的原始字符串
    cleaned_prompt = ''.join(char for char in prompt if ord(char) >= 32)
    return cleaned_prompt.strip()

async def save_result(result, json_filename, output_dir):
    """保存单个结果到JSON文件，使用原来的JSON文件名"""
    save_path = os.path.join(output_dir, json_filename)

    original_data = result.get('original_data', {})

    save_data = {
        'path': original_data.get('path', ''),
        'raw_label': original_data.get('label', ''),
        'text': original_data.get('text', ''),
        'prompt': extract_before_requirement(result.get('prompt_used', '')),
        'scene': result.get('content', ''),
        'process': result.get('reasoning', ''),
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


def find_json_files_in_range(json_dir, clarification_dir, output_dir, start=None, end=None):
    """遍历JSON目录，找到符合条件的JSON文件"""
    available_files = []
    t_json_files = []  # 存储 clarification 为 T 的 JSON 文件

    if not os.path.exists(json_dir):
        print(f"JSON目录不存在: {json_dir}")
        return available_files

    # 获取已存在输出目录中的文件名集合
    existing_output_files = set(os.path.splitext(f)[0] for f in os.listdir(output_dir) if f.lower().endswith('.json')) if os.path.exists(output_dir) else set()

    # 获取 clarification 目录中的 JSON 文件集合（使用完整文件名）
    clarification_files = set(f for f in os.listdir(clarification_dir) if f.lower().endswith('.json')) if os.path.exists(clarification_dir) else set()

    # 遍历目录中的所有JSON文件
    for filename in os.listdir(json_dir):
        if filename.lower().endswith('.json'):
            json_path = os.path.join(json_dir, filename)

            # 额外检查：检查对应的 clarification JSON 是否存在
            if filename not in clarification_files:
                print(f"警告：{filename} 在 clarification 目录中没有对应的文件")
                continue

            # 检查 clarification JSON 的 "clarification" 字段
            clarification_path = os.path.join(clarification_dir, filename)
            try:
                with open(clarification_path, 'r', encoding='utf-8') as f:
                    clarification_data = json.load(f)
                    if clarification_data.get('clarification') == 'T':
                        t_json_files.append((filename, filename))
                        continue  # 不添加到 available_files
            except Exception as e:
                print(f"读取 {filename} 的 clarification 失败: {e}")
                continue

            # 尝试从文件名提取数字
            try:
                # 假设文件名格式为 "数字.jpg.json"
                number_part = filename.split('.')[0]
                file_number = int(number_part)

                # 处理范围过滤和输出目录检查
                base_name = os.path.splitext(filename)[0]
                if (start is None or end is None or (start <= file_number <= end)) and \
                   base_name not in existing_output_files:
                    available_files.append((json_path, filename))

            except ValueError:
                # 如果无法提取数字，跳过该文件
                print(f"无法从文件名 {filename} 提取数字")
                continue

    # 打印 clarification 为 T 的 JSON 文件
    if t_json_files:
        print("Clarification 为 T 的 JSON 文件:")
        for idx, (base_name, filename) in enumerate(t_json_files, 1):
            print(f"序号 {idx}: {filename}")

    # 按文件名排序
    available_files.sort(key=lambda x: x[1])

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

    json_dir = str(random_dataset("raw_message"))
    meme_dir = str(random_dataset("meme"))
    feature_file = str(random_dataset("feature.txt"))
    clarification_dir = str(random_dataset("clarification"))

    # 检查目录是否存在
    if not os.path.exists(json_dir):
        print(f" JSON目录不存在: {json_dir}")
        return

    if not os.path.exists(meme_dir):
        print(f" Meme目录不存在: {meme_dir}")
        return

    if not os.path.exists(clarification_dir):
        print(f" Meme目录不存在: {clarification_dir}")
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

    # 创建输出目录
    output_dir = str(random_dataset(storage_file))
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
    available_files = find_json_files_in_range(json_dir,clarification_dir,output_dir,start, end)

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

    async def process_with_semaphore(json_path, filename,pool_manager):
        """带并发控制的处理函数"""
        async with semaphore:
            result = await process_single_json_with_thinking(json_path, meme_dir, client,storage_file,clarification_dir,pool_manager,feature_text)

            if result:
                await save_result(result, filename, output_dir)
                return filename
            return None

    # 创建进度条
    pbar = tqdm(total=len(valid_files), desc='处理文件', unit='个')

    # 并发处理所有文件
    tasks = []
    pool_manager = PoolManager()
    for json_path, filename in valid_files:
        task = asyncio.create_task(process_with_semaphore(json_path, filename,pool_manager))
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
    asyncio.run(main("harmless"))