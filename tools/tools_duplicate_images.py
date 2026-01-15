import argparse
import shutil
from pathlib import Path
'''
将【指定文件夹】 下面的文件 复制 为【num】 份 

'''


def parse_args():
    parser = argparse.ArgumentParser(
        description="复制目录下的图片文件，用于规模测试"
    )
    parser.add_argument(
        "--src",
        # required=True,
        default="data/buckets/batch_001/心脏超声诊断报告单", 
        help="源图片目录，例如：data/buckets/batch_001/心脏超声诊断报告单",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        #  required=True,
        default=100,  # 添加默认值
        help="目标图片总数量，例如：100 / 400 / 1000",
    )
    parser.add_argument(
        "--suffix",
        default="dup",
        help="复制文件的后缀标识，默认 dup",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    src_dir = Path(args.src)
    assert src_dir.exists(), f"目录不存在: {src_dir}"
    assert src_dir.is_dir(), f"不是目录: {src_dir}"

    images = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    assert images, f"目录中没有图片文件: {src_dir}"

    current_count = len(images)
    target = args.target_size

    print(f"[INFO] 当前图片数量: {current_count}")
    print(f"[INFO] 目标图片数量: {target}")

    if current_count >= target:
        print("[INFO] 当前数量已 >= 目标数量，无需复制")
        return

    needed = target - current_count
    print(f"[INFO] 需要新增复制文件数: {needed}")

    idx = 1
    created = 0

    while created < needed:
        for src_img in images:
            if created >= needed:
                break

            stem = src_img.stem
            suffix = src_img.suffix
            new_name = f"{stem}__{args.suffix}{idx:04d}{suffix}"
            dst_img = src_dir / new_name

            if dst_img.exists():
                idx += 1
                continue

            shutil.copy2(src_img, dst_img)
            created += 1
            idx += 1

    print(f"[DONE] 实际新增文件数: {created}")
    print(f"[DONE] 当前目录总文件数: {len(list(src_dir.iterdir()))}")


if __name__ == "__main__":
    main()
