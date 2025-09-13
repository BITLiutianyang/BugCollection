from urllib.parse import urlparse
import requests
import traceback
import re
import json
from method_extractor import JavaMethodExtractor
import time
import tiktoken
import patch_prompts

from llm_request import do_request
# import prompts.patch_prompts as patch_prompts
# import prompts.message_prompts as message_prompts

GITHUB_TOKEN = "your githubtoken"
def LLM_analyze(description_pro, patch_list, method_body_list):
    system_prompt = patch_prompts.system_analyze_prompt
    patch = ""
    user_prompt = ""  # 初始化 user_prompt
    
    if len(patch_list) == 0:
        return None
    
    if len(method_body_list) != 0:
        for i in range(len(patch_list)):
            patch_body = patch_list[i]
            method_body = method_body_list[i]
            patch = patch + f"\n\nPatch[{str(i)}]:\n" + patch_body + f"\n\nMethod[{str(i)}]:\n" + method_body + "\n\n"
            user_prompt = patch_prompts.user_analyze_prompt.format(description = description_pro, patches = patch)
    else:
        for i in range(len(patch_list)):
            patch_body = patch_list[i]
            patch = patch + f"\n\nPatch[{str(i)}]:\n" + patch_body + "\n\n"
            user_prompt = patch_prompts.user_analyze_prompt.format(description = description_pro, patches = patch)
            
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}]
    response = do_request(messages)
    
    messages.extend([{"role": "assistant", "content": response},
                    {"role": "user", "content": patch_prompts.json_prompt}])
    
    response = do_request(messages)
    data_json = to_json(response)
    return data_json

                        
def get_line(patch):
    lines = patch.splitlines()
    pattern = r'[-+]?\d+,\d+'
    pattern2 = r'[-+]\d+,\d+'
    modify = []
    for line in lines:
        match = 0

        if re.search(pattern, line):
            match = 1
        if match == 1:
            parts = line.split()
            for part in parts:
                if re.match(pattern2, part):
                    number, column = part.split(',')
                    sign = '-' if part[0] == '-' else '+'
                    if sign == '-':
                        patch_start_line = abs(int(number))
                        patch_start_line = patch_start_line
                        modify.append(patch_start_line)
    return modify                        
def to_json(response):
    # 定义正则表达式模式,匹配json字符串
    patterns = [
        r'```json\s*(.*?)\s*```',  # 优先匹配json代码块
        r'```\s*(.*?)\s*```',      # 其次匹配任意代码块
        r'({.*})'                  # 最后贪婪匹配整个JSON对象
    ]
    
    json_txt = None
    for pattern in patterns:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            json_txt = match.group(1).strip()

        if json_txt:
            try:
                json_response = json.loads(json_txt)
                return json_response
            except json.JSONDecodeError as e:
                print(f"JSON 解码错误: {e}")
                print("response: ", response)
                return None
    else:
        return None
    
        

def get_method_inline(diff_line):
    if not diff_line.startswith("@@") or " -" not in diff_line or " +" not in diff_line:
        raise ValueError("输入的diff字符串格式不正确")
    old_file_info = diff_line.split(" -")[1].split(" +")[0]
    start_line, num_lines = old_file_info.split(",")  # 分割为起始行号和行数
    start_line = int(start_line)
    num_lines = int(num_lines) 
    
    return start_line, start_line + num_lines

def get_method(start_line, end_line, method_lines, method_bodies):
    methods = []
    methods_key = []
    for method in method_lines:
        method_num = method_lines[method]
        sline = method_num[0]
        eline = method_num[1]

        if sline <= end_line and start_line <= eline:
            methods.append(method_bodies[method])
            methods_key.append(method)
            break
    return methods, methods_key

def process_(record_dict, description_pro):
    
    files = record_dict['files']
    # 这里如果要是爬虫的时候，需要更改哦
    file0 = files[0]
    before_file_content = file0['old_content']  
    after_file_content = file0['new_content']
            
    patches = file0['patch']
    
    modify_lines = get_line(patches)
    
    patches_lines = patches.split('\n')
    
    diff_lines = []
    diff_line_2_patch_list = {}
    diff_line_2_patch_str = {}
    sindex = 0
    for index, patch_line in enumerate(patches_lines):
        patch_line = patch_line.strip()
        if patch_line.startswith('@@'):
            diff_lines.append(patch_line)
        if patch_line.startswith('@@') or index == (len(patches_lines) - 1):
            if sindex < index:
                diff_line_2_patch_list[patches_lines[sindex]] = patches_lines[sindex:index]
                diff_line_2_patch_str[patches_lines[sindex]] = '\n'.join(patches_lines[sindex:index])
                
            sindex = index
    try:
        before_java_extractor = JavaMethodExtractor(before_file_content)
        after_java_extractor = JavaMethodExtractor(after_file_content)
    except Exception as e:
        traceback.print_exc()
        return None, None, None
        
    before_method_bodies = before_java_extractor.get_method_bodies()
    before_method_lines = before_java_extractor.get_method_lines()
    
    after_method_bodies = after_java_extractor.get_method_bodies()
    after_method_lines = after_java_extractor.get_method_lines()
    
    
    patch_infor = {}
    patch_list_all = []
    for diff_line in diff_line_2_patch_list:
        start_line, end_line = get_method_inline(diff_line)
        patch = diff_line_2_patch_str[diff_line]
        patch_list_all.append(patch)
        method_bodys, method_keys = get_method(start_line, end_line, before_method_lines, before_method_bodies)
        
        if len(method_bodys) == 0:
            continue

        method_name = method_keys[0][0]
        method_key = method_keys[0]
        method_body = method_bodys[0]
        
        if method_key not in patch_infor:
            patch_infor[method_key] = {}
        if "diff_line" not in patch_infor[method_key]:
            patch_infor[method_key]["diff_line"] = {} 

        # patch_infor[method_key]["diff_line"]["patch_summary"] = patch_summary
        patch_infor[method_key]["diff_line"]["method_name"] = method_name
        patch_infor[method_key]["diff_line"]["method_body"] = method_body
        patch_infor[method_key]["diff_line"]["patch"] = patch

    
    method_body_list = []
    patch_list = []
    for method_key in patch_infor:
        for diff_line in patch_infor[method_key]:
            # patch_summary = patch_infor[method_key][diff_line]["patch_summary"]
            method_name = patch_infor[method_key][diff_line]["method_name"]
            method_body = patch_infor[method_key][diff_line]["method_body"]
            patch = patch_infor[method_key][diff_line]["patch"]
            patch_list.append(patch)
            method_body_list.append(method_body)
    
    if patch_infor == {}:
        patch_list = patch_list_all

    # func_token = len(encoding.encode("\n".join(method_body_list)))
    func_token = len("\n".join(method_body_list))
    
    print("patch_list: ", len(patch_list))
     
    if func_token > 1000:
        response = LLM_analyze(description_pro, patch_list, [])
    else:
        response = LLM_analyze(description_pro, patch_list, method_body_list)
            
    # print("response: ", response)
    return response
