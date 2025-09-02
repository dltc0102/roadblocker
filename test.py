import requests, re

ENGLISH_ROAD_TYPES = ["Street", "Road", "Drive", "Terrace", "Highway", "Tunnel", "Expressway", "Bypass", "Avenue", "Boulevard"]
CHINESE_ROAD_TYPES = ["街道", "街", "隧道", "道", "公路"]

def get_gm_api_key() -> str:
    with open("apikey.txt", 'r', encoding='utf-8') as key_f:
        return key_f.readline().strip()

def regulate_dashes(name_str: str) -> str:
    if not name_str: return name_str
    dash_chars = '\u002D\u2010\u2011\u2012\u2013\u2014\u2015\u2053\u207B\u208B\u2212\u2E17\u2E1A\u2E3A\u2E3B\uFE58\uFE63\uFF0D\uFF5E'
    dash_translation = str.maketrans(dash_chars, '-' * len(dash_chars))
    name_str = name_str.translate(dash_translation)
    return re.sub(r'\s*-\s*', '-', name_str)

def is_valid_road_eng(name_str: str) -> bool:
    return any(word in name_str for word in ENGLISH_ROAD_TYPES)

def is_valid_road_cn(name_str: str) -> bool:
    return any(word in name_str for word in CHINESE_ROAD_TYPES)

def is_text_chinese(text: str) -> bool:
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
    return bool(chinese_pattern.search(text))

def get_lang_names_v4(name_str: str) -> tuple[str, str]:
    dash_name_str = regulate_dashes(name_str)
    dash_positions = [i for i, char in enumerate(dash_name_str) if char == '-']

    if not dash_positions:
        chinese_matches = re.findall(r'[\u4e00-\u9fff]+', dash_name_str)
        chinese_name = ' '.join(chinese_matches) if chinese_matches else ""
        english_name = re.sub(r'[\u4e00-\u9fff]+', '', dash_name_str).strip()
        return english_name, chinese_name

    space_positions = [i for i, char in enumerate(dash_name_str) if char == ' ']
    first_char = dash_name_str[0]
    left_is_chinese = is_text_chinese(first_char)

    english_part = None
    chinese_part = None
    if left_is_chinese:
        # right is english
        # 青山公路-青龍頭段 Castle Peak Road-Tsing Lung Tau
        chinese_part = dash_name_str[:space_positions[0]].strip()
        english_part = dash_name_str[space_positions[0]+1:].strip()
    else:
        # left is english
        # Castle Peak Road-Tsing Lung Tau 青山公路-青龍頭段
        chinese_part = dash_name_str[space_positions[-1]:].strip()
        english_part = dash_name_str[:space_positions[-1]].strip()

    return extract_english_name(english_part), extract_chinese_name(chinese_part)

def extract_english_name(text: str) -> str:
    if not text: return ""
    if '-' in text:
        parts = [part.strip() for part in text.split('-')]
        for part in parts:
            if any(road_type in part for road_type in ENGLISH_ROAD_TYPES): return part
            if part and part[0].isascii(): return part
        return parts[0] if parts else ""

    return text.strip()

def extract_chinese_name(text: str) -> str:
    if not text: return ""
    if '-' in text:
        parts = [part.strip() for part in text.split('-')]
        for part in parts:
            if any(road_type in part for road_type in CHINESE_ROAD_TYPES): return part
            if any('\u4e00' <= char <= '\u9fff' for char in part): return part
        return parts[0] if parts else ""
    return text.strip()

problems = [
    "青山公路－青龍頭段 Castle Peak Road – Tsing Lung Tau",
    "Castle Peak Road – Tsing Lung Tau 青山公路－青龍頭段",
    # "Stanley Main Street 赤柱大街",
    # "赤柱大街 Stanley Main Street",
]
def main():
    for problem in problems:
        a1 = get_lang_names_v4(problem)
        print(a1)
        print()
if __name__ == "__main__":
    main()