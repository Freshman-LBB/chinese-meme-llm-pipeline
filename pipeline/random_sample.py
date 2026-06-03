import os
import traceback  # 添加这个导入
import random
import shutil
import secrets
from pathlib import Path
import argparse
import time

def check_and_clean_target_directory(target_dir):
    """检查目标目录并询问是否清空"""
    if target_dir.exists():
        existing_files = list(target_dir.glob('*.jpg'))
        if existing_files:
            print(f"目标目录 {target_dir} 已包含 {len(existing_files)} 张图片")
            while True:
                choice = input("是否清空目录？(y/n): ").lower()
                if choice == 'y':
                    shutil.rmtree(target_dir)
                    target_dir.mkdir(parents=True, exist_ok=True)
                    print(f"已清空目录: {target_dir}")
                    return False, 0  # 返回False表示目录被清空，0表示现有图片数量
                elif choice == 'n':
                    return True, len(existing_files)  # 返回True表示保留现有数据，并返回现有图片数量
        else:
            target_dir.mkdir(parents=True, exist_ok=True)
    else:
        target_dir.mkdir(parents=True, exist_ok=True)
    return False, 0

def count_existing_images(target_dir):
    """统计目标目录已有图片数量"""
    return len(list(target_dir.glob('*.jpg')))


def secure_random_sample(start, end, count):
    if count > (end - start + 1):
        raise ValueError(f"请求的数量 {count} 超过了范围 [{start}, {end}] 的大小")

    # 方法1: 使用secrets模块（密码学安全的随机数）
    random.seed(secrets.randbits(128))

    # 方法2: 结合当前时间的微秒作为额外的随机种子
    time.sleep(0.001)  # 确保时间变化
    random.seed(random.getstate()[1][0] ^ int(time.time() * 1000000))

    # 生成范围内的所有数字
    all_numbers = list(range(start, end + 1))

    # 使用Fisher-Yates洗牌算法 + secrets随机数
    for i in range(len(all_numbers) - 1, 0, -1):
        j = secrets.randbelow(i + 1)
        all_numbers[i], all_numbers[j] = all_numbers[j], all_numbers[i]

    # 再次使用random.sample作为双重保险
    random.shuffle(all_numbers)

    return all_numbers[:count]


def clear_target_directory(target_dir):
    """清空目标目录"""
    if target_dir.exists():
        print(f"清空目标目录: {target_dir}")
        shutil.rmtree(target_dir)

    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"已创建目录: {target_dir}")


def find_and_copy_images_with_fallback(selected_numbers, source_dir, target_dir, target_count, max_range):

    if not source_dir.exists():
        print(f"错误：源目录不存在 {source_dir}")
        return 0, 0, []

    found_count = 0
    attempted_numbers = set()
    successful_numbers = []
    failed_attempts = []

    print(f"\n开始复制图片...")
    print(f"源目录: {source_dir}")
    print(f"目标目录: {target_dir}")
    print(f"目标数量: {target_count}")

    # 将选中的数字转换为队列
    number_queue = list(selected_numbers)
    current_index = 0

    while found_count < target_count and current_index < len(number_queue):
        number = number_queue[current_index]

        # 如果这个数字已经尝试过，跳过
        if number in attempted_numbers:
            current_index += 1
            continue

        attempted_numbers.add(number)
        image_filename = f"{number}.jpg"
        source_file = source_dir / image_filename
        target_file = target_dir / image_filename

        if source_file.exists():
            try:
                shutil.copy2(source_file, target_file)
                print(f"✓ 已复制: {image_filename}")
                found_count += 1
                successful_numbers.append(number)
            except Exception as e:
                print(f"✗ 复制失败 {image_filename}: {e}")
                failed_attempts.append(number)
        else:
            print(f"✗ 未找到: {image_filename}, 尝试下一个数字...")
            failed_attempts.append(number)

            # 如果文件不存在，尝试下一个数字
            next_number = number + 1
            while next_number <= max_range and next_number in attempted_numbers:
                next_number += 1

            if next_number <= max_range:
                number_queue.append(next_number)
                print(f"  → 添加候补数字: {next_number}")

        current_index += 1

    # 如果还没有达到目标数量，随机添加更多数字
    while found_count < target_count:
        # 随机生成一个新的数字
        attempts = 0
        max_attempts = 1000

        while attempts < max_attempts:
            random_number = random.randint(1, max_range)
            if random_number not in attempted_numbers:
                attempted_numbers.add(random_number)
                image_filename = f"{random_number}.jpg"
                source_file = source_dir / image_filename
                target_file = target_dir / image_filename

                if source_file.exists():
                    try:
                        shutil.copy2(source_file, target_file)
                        print(f"✓ 已复制 (补充): {image_filename}")
                        found_count += 1
                        successful_numbers.append(random_number)
                        break
                    except Exception as e:
                        print(f"✗ 复制失败 {image_filename}: {e}")
                        failed_attempts.append(random_number)
                else:
                    failed_attempts.append(random_number)

            attempts += 1

        if attempts >= max_attempts:
            print(f"警告：经过 {max_attempts} 次尝试仍无法找到足够的图片文件")
            break

    return found_count, len(failed_attempts), successful_numbers


def get_user_input():
    """获取用户输入参数"""
    parser = argparse.ArgumentParser(description='随机选择并复制图片文件')
    parser.add_argument('--start', type=int, help='起始数字')
    parser.add_argument('--end', type=int, help='结束数字')
    parser.add_argument('--count', type=int, help='选择数量')
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='交互式输入模式')

    args = parser.parse_args()

    # 如果没有提供参数或者指定了交互模式，则使用交互式输入
    if args.interactive or (args.start is None or args.end is None or args.count is None):
        print("=== 随机图片选择工具 ===")
        print("请输入参数：")

        while True:
            try:
                start = int(input("起始数字: "))
                end = int(input("结束数字: "))
                if start > end:
                    print("起始数字不能大于结束数字，请重新输入")
                    continue
                count = int(input("选择数量: "))
                if count <= 0:
                    print("选择数量必须大于0，请重新输入")
                    continue
                break
            except ValueError:
                print("请输入有效的整数")
    else:
        start, end, count = args.start, args.end, args.count

        # 验证参数
        if start > end:
            raise ValueError("起始数字不能大于结束数字")
        if count <= 0:
            raise ValueError("选择数量必须大于0")

    return start, end, count


def main():
    try:
        # 获取用户输入
        start, end, count = get_user_input()

        print(f"\n=== 参数确认 ===")
        print(f"数字范围: [{start}, {end}]")
        print(f"选择数量: {count}")
        print(f"总范围大小: {end - start + 1}")

        # 定义路径
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from config.paths import DATA_ROOT, random_dataset

        source_dir = DATA_ROOT / "meme"
        target_dir = random_dataset("meme")

        # 修改这里，获取是否保留现有数据和现有图片数量
        try:
            keep_existing, existing_count = check_and_clean_target_directory(target_dir)
        except Exception as e:
            print(f"检查目录时发生错误: {e}")
            print(traceback.format_exc())
            return 1

        if keep_existing:
            print(f"当前已有 {existing_count} 张图片")
            # 如果已有图片数量超过或等于用户指定数量，询问是否继续
            if existing_count >= count:
                choice = input(f"已有 {existing_count} 张图片，是否仍要继续随机选择？(y/n): ").lower()
                if choice != 'y':
                    print("操作已取消")
                    return 0

            # 调整目标数量，补充到指定数目
            count = count - existing_count
            if count <= 0:
                print("无需添加更多图片")
                return 0

        # 如果保留现有数据，不清空目录
        if not keep_existing:
            try:
                clear_target_directory(target_dir)
            except Exception as e:
                print(f"清空目录时发生错误: {e}")
                print(traceback.format_exc())
                return 1

        print(f"\n=== 路径信息 ===")
        print(f"源目录: {source_dir}")
        print(f"目标目录: {target_dir}")

        # 检查源目录是否存在
        if not source_dir.exists():
            print(f"错误：源目录不存在 {source_dir}")
            return 1

        # 统计源目录中可用的图片
        try:
            available_images = list(source_dir.glob("*.jpg"))
            available_numbers = []
            for img in available_images:
                try:
                    number = int(img.stem)
                    available_numbers.append(number)
                except ValueError:
                    continue
        except Exception as e:
            print(f"统计图片时发生错误: {e}")
            print(traceback.format_exc())
            return 1

        print(f"源目录中总共有 {len(available_numbers)} 张图片")

        if len(available_numbers) < count:
            print(f"警告：可用图片总数 ({len(available_numbers)}) 少于请求数量 ({count})")
            print(f"将尽力复制最多 {len(available_numbers)} 张图片")

        if len(available_numbers) == 0:
            print("没有可复制的图片")
            return 1

        # 扩大搜索范围到所有可用图片的最大值
        max_available = max(available_numbers) if available_numbers else end
        search_range = max(end, max_available)

        # 随机选择数字（从用户指定的范围开始）
        print(f"\n=== 开始随机选择 ===")
        initial_count = min(count * 2, end - start + 1)  # 选择更多数字作为候选

        try:
            selected_numbers = secure_random_sample(start, end, initial_count)
        except Exception as e:
            print(f"随机选择数字时发生错误: {e}")
            print(traceback.format_exc())
            return 1

        print(f"初始随机选中的数字: {sorted(selected_numbers[:20])}{'...' if len(selected_numbers) > 20 else ''}")

        # 复制图片（带回退机制）
        try:
            found_count, failed_count, successful_numbers = find_and_copy_images_with_fallback(
                selected_numbers, source_dir, target_dir, count, search_range
            )
        except Exception as e:
            print(f"复制图片时发生错误: {e}")
            print(traceback.format_exc())
            return 1

        # 输出统计信息
        print(f"\n=== 复制完成 ===")
        print(f"目标数量: {count}")
        print(f"实际复制: {found_count}")
        print(f"成功率: {found_count / count * 100:.1f}%")

        if found_count == count:
            print("✅ 已成功达到目标数量！")
        elif found_count < count:
            print(f"⚠️  未能达到目标数量，缺少 {count - found_count} 张")

        print(f"\n实际复制的数字: {sorted(successful_numbers)}")
        print(f"\n所有文件已保存到: {target_dir}")

    except Exception as e:
        print(f"发生未处理的错误: {e}")
        print(traceback.format_exc())
        return 1

    return 0


if __name__ == "__main__":
    result = main()