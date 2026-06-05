import argparse
import os
import re
import sys
import time
from html import unescape
from pathlib import Path

import requests

try:
    import gdown
except ImportError as exc:
    raise SystemExit("请先安装依赖：pip install gdown requests") from exc


DEFAULT_CVA_URL = "https://drive.google.com/drive/folders/1NGXnVG0gUFHfDcUvJxfozCgiI4WwquVk"
FILE_RE = re.compile(r"https://drive\.google\.com/file/d/([-\w]{25,})/view\?usp=drive_web")
FOLDER_RE = re.compile(r"https://drive\.google\.com/drive/folders/([-\w]{25,})")
TITLE_RE = re.compile(r"<title>(.*?)</title>", re.S)


def unique_keep_order(items):
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def extract_folder_id(value: str) -> str:
    value = value.strip()
    match = re.search(r"folders/([-\w]{25,})", value)
    if match:
        return match.group(1)
    if re.fullmatch(r"[-\w]{25,}", value):
        return value
    raise ValueError(f"无法解析 Google Drive 文件夹 ID: {value}")


def sanitize_name(name: str) -> str:
    name = unescape(name).strip()
    name = re.sub(r'[\\/*?:"<>|]', "_", name)
    return name or "unnamed_folder"


def fetch_embedded_folder_html(
    session: requests.Session,
    folder_id: str,
    retries: int,
    retry_wait: int,
) -> str:
    url = f"https://drive.google.com/embeddedfolderview?id={folder_id}#list"
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(
                url,
                timeout=(15, 60),
                headers={"User-Agent": "Mozilla/5.0"},
            )
            resp.raise_for_status()
            return resp.text
        except Exception as exc:
            if attempt == retries:
                raise RuntimeError(f"获取文件夹页面失败，folder_id={folder_id}") from exc
            wait_time = retry_wait * attempt
            print(f"[WARN] folder_id={folder_id} 第 {attempt}/{retries} 次获取失败：{exc}")
            print(f"[INFO] {wait_time} 秒后重试...")
            time.sleep(wait_time)


def parse_folder_title(html_text: str, folder_id: str) -> str:
    match = TITLE_RE.search(html_text)
    if not match:
        return folder_id
    title = match.group(1)
    title = title.replace(" - Google 云端硬盘", "").replace(" - Google Drive", "")
    return sanitize_name(title)


def parse_file_ids(html_text: str):
    return unique_keep_order(FILE_RE.findall(html_text))


def parse_subfolder_ids(html_text: str, current_folder_id: str):
    folder_ids = unique_keep_order(FOLDER_RE.findall(html_text))
    return [folder_id for folder_id in folder_ids if folder_id != current_folder_id]


def load_completed_ids(state_file: Path):
    if not state_file.exists():
        return set()
    return {
        line.strip()
        for line in state_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def mark_completed(state_file: Path, file_id: str):
    with state_file.open("a", encoding="utf-8") as f:
        f.write(file_id + "\n")


def download_file(file_id: str, target_dir: Path, retries: int, retry_wait: int):
    target_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}"
    output = str(target_dir) + os.sep

    for attempt in range(1, retries + 1):
        try:
            result = gdown.download(
                url=url,
                output=output,
                quiet=False,
                resume=True,
                fuzzy=True,
            )
            if result:
                return
            raise RuntimeError("gdown 未返回输出文件路径")
        except Exception as exc:
            if attempt == retries:
                raise RuntimeError(f"下载失败，file_id={file_id}") from exc
            print(f"[WARN] file_id={file_id} 第 {attempt}/{retries} 次失败：{exc}")
            print(f"[INFO] {retry_wait} 秒后重试...")
            time.sleep(retry_wait)


def recursive_download(
    session: requests.Session,
    folder_id: str,
    parent_dir: Path,
    visited: set,
    retries: int,
    retry_wait: int,
    completed_ids: set,
    state_file: Path,
    resume_tracker: dict,
):
    if folder_id in visited:
        return
    visited.add(folder_id)

    html_text = fetch_embedded_folder_html(
        session=session,
        folder_id=folder_id,
        retries=retries,
        retry_wait=retry_wait,
    )
    folder_name = parse_folder_title(html_text, folder_id)
    current_dir = parent_dir / folder_name
    current_dir.mkdir(parents=True, exist_ok=True)

    file_ids = parse_file_ids(html_text)
    subfolder_ids = parse_subfolder_ids(html_text, folder_id)

    print(f"[INFO] 处理文件夹：{current_dir}")
    print(f"[INFO] 发现文件 {len(file_ids)} 个，子文件夹 {len(subfolder_ids)} 个")

    for idx, file_id in enumerate(file_ids, start=1):
        if not resume_tracker["started"]:
            if file_id != resume_tracker["target_file_id"]:
                print(f"[INFO] 跳过恢复点之前的文件 {idx}/{len(file_ids)}: {file_id}")
                continue
            resume_tracker["started"] = True
            print(f"[INFO] 已定位恢复点 {idx}/{len(file_ids)}: {file_id}")

        if file_id in completed_ids:
            print(f"[INFO] 跳过已完成文件 {idx}/{len(file_ids)}: {file_id}")
            continue

        print(f"[INFO] 下载文件 {idx}/{len(file_ids)}: {file_id}")
        download_file(file_id, current_dir, retries=retries, retry_wait=retry_wait)
        completed_ids.add(file_id)
        mark_completed(state_file, file_id)

    for subfolder_id in subfolder_ids:
        recursive_download(
            session=session,
            folder_id=subfolder_id,
            parent_dir=current_dir,
            visited=visited,
            retries=retries,
            retry_wait=retry_wait,
            completed_ids=completed_ids,
            state_file=state_file,
            resume_tracker=resume_tracker,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        default=DEFAULT_CVA_URL,
        help="Google Drive 文件夹链接或 folder id，默认是 CVA 根目录",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "CVA"),
        help="下载输出目录",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=10,
        help="单文件失败后的最大重试次数",
    )
    parser.add_argument(
        "--retry-wait",
        type=int,
        default=30,
        help="重试等待秒数",
    )
    parser.add_argument(
        "--start-file-id",
        default=None,
        help="从指定 file_id 开始继续下载，适合中途中断后恢复",
    )
    args = parser.parse_args()

    folder_id = extract_folder_id(args.folder)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    state_file = output_dir / ".downloaded_file_ids.txt"
    completed_ids = load_completed_ids(state_file)

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    visited = set()
    resume_tracker = {
        "target_file_id": args.start_file_id,
        "started": args.start_file_id is None,
    }

    print(f"[INFO] 根文件夹 ID: {folder_id}")
    print(f"[INFO] 输出目录: {output_dir}")
    print(f"[INFO] 已记录完成文件数: {len(completed_ids)}")

    recursive_download(
        session=session,
        folder_id=folder_id,
        parent_dir=output_dir,
        visited=visited,
        retries=args.retries,
        retry_wait=args.retry_wait,
        completed_ids=completed_ids,
        state_file=state_file,
        resume_tracker=resume_tracker,
    )

    if args.start_file_id and not resume_tracker["started"]:
        sys.exit(f"未找到 start-file-id: {args.start_file_id}")

    print("[INFO] 下载完成")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\n已手动中断下载")
    except Exception as exc:
        sys.exit(f"\n下载失败: {exc}")