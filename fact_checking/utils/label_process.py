import re
from typing import List


# <tag>text</tag> 提取 text
def extract_label(tag: str, text: str = "", default: str = "") -> str:
    pattern = rf"<" + tag + r">(.*?)</" + tag + r">"
    m = re.search(pattern, text, re.S | re.I)
    return m.group(1).strip() if m else default


# <tag>text</tag> 提取所有 text
def extract_all_label(tag: str, text: str = "", default: List[str] = None) -> List[str]:
    if default is None:
        default = []
    pattern = rf"<{tag}>(.*?)</{tag}>"
    matches = re.findall(pattern, text, re.S | re.I)
    return [m.strip() for m in matches] if matches else default
