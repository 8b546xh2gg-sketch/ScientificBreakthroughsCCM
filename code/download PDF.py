import os
import re
import time
import mimetypes
from pathlib import Path
from urllib.parse import urlparse, unquote
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
from tqdm import tqdm

# 配置：适当调整超时、重试等
REQUEST_TIMEOUT = (5, 120)  # (connect, read) seconds
MAX_RETRIES = 3
CHUNK_SIZE = 8192
MAX_FILENAME_LEN = 200  # 防止过长文件名导致问题

def sanitize_filename(name: str) -> str:
    if not isinstance(name, str):
        name = str(name)
    # 去掉非法 Windows 字符并压缩空白
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name).strip()
    name = re.sub(r'\s+', ' ', name)
    if len(name) > MAX_FILENAME_LEN:
        name = name[:MAX_FILENAME_LEN]
    return name or "untitled"

def extension_from_url(url: str) -> str:
    try:
        path = unquote(urlparse(url).path)
        base, ext = os.path.splitext(path)
        if ext and len(ext) <= 10:
            return ext
    except Exception:
        pass
    return ''

def extension_from_content_disposition(cd: str) -> str:
    if not cd:
        return ''
    m = re.search(r'filename\*?=(?:UTF-8\'\')?["\']?([^;"\']+)', cd, flags=re.IGNORECASE)
    if m:
        _, ext = os.path.splitext(unquote(m.group(1)))
        return ext
    return ''

def extension_from_content_type(ct: str) -> str:
    if not ct:
        return ''
    # 去除参数
    ct_main = ct.split(';')[0].strip()
    ext = mimetypes.guess_extension(ct_main)
    if ext == '.jpe':  # mimetypes oddity
        ext = '.jpg'
    return ext or ''

def determine_extension(url: str, resp: requests.Response) -> str:
    # 1. 从 URL
    ext = extension_from_url(url)
    if ext:
        return ext
    # 2. 从 Content-Disposition
    cd = resp.headers.get('content-disposition', '')
    ext = extension_from_content_disposition(cd)
    if ext:
        return ext
    # 3. 从 Content-Type
    ct = resp.headers.get('content-type', '')
    ext = extension_from_content_type(ct)
    if ext:
        return ext
    # 4. 默认二进制
    return '.bin'

def make_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(total=MAX_RETRIES, backoff_factor=0.5,
                    status_forcelist=(500, 502, 503, 504), allowed_methods=frozenset(['GET', 'POST']))
    session.mount('http://', HTTPAdapter(max_retries=retries))
    session.mount('https://', HTTPAdapter(max_retries=retries))
    # 常用 UA，避免被部分服务器拒绝
    session.headers.update({'User-Agent': 'Mozilla/5.0 (compatible; downloader/1.0)'})
    return session

def main():
    base_dir = Path(__file__).resolve().parent
    excel_path = base_dir / 'medicine data_extracted292.xlsx'
    if not excel_path.exists():
        print(f"找不到 Excel 文件: {excel_path}")
        return

    out_dir = base_dir / 'papers_collection'
    out_dir.mkdir(parents=True, exist_ok=True)

    # 读取 Excel（使用 openpyxl 引擎）
    df = pd.read_excel(excel_path, engine='openpyxl')

    # 确认必要列
    if 'extracted_hrefs' not in df.columns or 'title' not in df.columns:
        print("Excel 文件中缺少 'extracted_hrefs' 或 'title' 列。")
        return

    # 新增列用于记录状态
    status_col = 'download_status'
    path_col = 'download_path'
    err_col = 'download_error'
    df[status_col] = ''
    df[path_col] = ''
    df[err_col] = ''

    session = make_session()
    total = len(df)
    success_count = 0
    fail_count = 0

    for idx in tqdm(range(total), desc="Downloading"):
        url = df.at[idx, 'extracted_hrefs']
        title = df.at[idx, 'title']
        # 如果 URL 为空，直接跳过（标记为 skipped），不计为失败
        if pd.isna(url) or not str(url).strip():
            df.at[idx, status_col] = 'skipped'
            df.at[idx, err_col] = ''
            continue

        try:
            with session.get(str(url), stream=True, timeout=REQUEST_TIMEOUT) as resp:
                resp.raise_for_status()

                # 强制以 PDF 保存（根据要求：要下载的文件是 PDF 文件）
                # 仍保留对真实响应头的判断用于日志或将来扩展
                ext = '.pdf'

                safe_name = sanitize_filename(title)
                # 确保文件名不以 .pdf 重复结尾
                if safe_name.lower().endswith('.pdf'):
                    safe_name = safe_name[:-4].rstrip()

                filename = f"{safe_name}{ext}"
                filepath = out_dir / filename

                # 避免同名覆盖：如果存在则追加 suffix
                counter = 1
                while filepath.exists():
                    filename = f"{safe_name}__{counter}{ext}"
                    filepath = out_dir / filename
                    counter += 1

                # 写入文件（逐块）
                with open(filepath, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)

                df.at[idx, status_col] = 'success'
                df.at[idx, path_col] = str(filepath)
                df.at[idx, err_col] = ''
                success_count += 1

                # 可选短暂等待，避免触发速率限制
                time.sleep(0.1)

        except Exception as e:
            df.at[idx, status_col] = 'failed'
            df.at[idx, err_col] = repr(e)
            fail_count += 1

    # 保存结果为新的 Excel 文件（防止覆盖原表）
    out_excel = excel_path.with_name(excel_path.stem + '_download_status.xlsx')
    df.to_excel(out_excel, index=False, engine='openpyxl')
    print(f"下载完成：成功 {success_count}，失败 {fail_count}。结果已保存到 {out_excel}")

if __name__ == '__main__':
    main()