# 专门用来生成message描述

from llm_analyzer import process4description
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os

def process_file(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as r:
        content = r.readlines()
        results = []
        for idx, content_line in enumerate(content):
            data = json.loads(content_line)

            description, description_pro, comments_url, comment_count = process4description(data)
            print(idx)
            data['description_pro_new'] = description_pro
            data['comments_url'] = comments_url
            data['comment_count'] = comment_count
            with open(output_path, "a", encoding="utf-8") as wf:
                wf.write(json.dumps(data) + '\n')
def main():  
    in_path = ""
    out_path = ""
    
    process_file(in_path, out_path)
                        
if __name__ == "__main__":
    main()
