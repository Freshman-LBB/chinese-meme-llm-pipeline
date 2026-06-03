import os
from pathlib import Path


def main():
    # root_dir = 项目根目录
    script_path = Path(__file__).resolve()
    root_dir = script_path.parents[2]

    harmful_dir = root_dir / "random" / "dataset" / "harmful"
    harmless_dir = root_dir / "random" / "dataset" / "harmless"
    meme_dir = root_dir / "random" / "dataset" / "meme"

    # 1. 获取 harmful / harmless 下所有 .jpg.json 文件名（只比对文件名）
    harmful_files = sorted(
        f.name
        for f in harmful_dir.glob("*.jpg.json")
        if f.is_file()
    )
    harmless_files = sorted(
        f.name
        for f in harmless_dir.glob("*.jpg.json")
        if f.is_file()
    )

    # 集合对比
    harmful_set = set(harmful_files)
    harmless_set = set(harmless_files)

    print("=== Step 1: 比对 harmful 和 harmless 的 json 文件名 ===")
    print(f"harmful json 文件数量: {len(harmful_files)}")
    print(f"harmless json 文件数量: {len(harmless_files)}")

    if harmful_set == harmless_set:
        print("✅ harmful 与 harmless 目录中的 .jpg.json 文件名集合完全一致。")
    else:
        print("❌ 文件名集合不一致：")
        only_in_harmful = sorted(harmful_set - harmless_set)
        only_in_harmless = sorted(harmless_set - harmful_set)

        if only_in_harmful:
            print("  仅在 harmful 中存在的文件：")
            for name in only_in_harmful:
                print("   -", name)

        if only_in_harmless:
            print("  仅在 harmless 中存在的文件：")
            for name in only_in_harmless:
                print("   -", name)

    # 2. 检查所有 json 对应的 jpg 是否都在 meme 目录内
    print("\n=== Step 2: 检查对应的图片是否存在于 random/dataset/meme ===")

    # 这里采用交集（如果你只信任其中一个目录，也可以只从 harmful_files 或 harmless_files 取）
    common_files = sorted(harmful_set & harmless_set)

    missing_images = []
    for json_name in common_files:
        # json_name 格式：数字.jpg.json -> 对应图片：数字.jpg
        jpg_name = json_name.replace(".jpg.json", ".jpg")
        jpg_path = meme_dir / jpg_name
        if not jpg_path.is_file():
            missing_images.append(jpg_name)

    if not missing_images:
        print("✅ 所有 json 对应的 .jpg 图片都能在 meme 目录中找到。")
    else:
        print("❌ 下列图片在 meme 目录中未找到：")
        for name in missing_images:
            print("   -", name)


if __name__ == "__main__":
    main()