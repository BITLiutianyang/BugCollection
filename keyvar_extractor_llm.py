import openai
import json
import os
from prompt_cv import user_cv_prompt_nocontent, system_cv_prompt
from llm_request import do_request
import re



class LLMInference:
    # def __init__(self, message, cv_candidate, static_report, llm_report, patch, method):
    def __init__(self, message, cv_candidate,  patch):
        self.message = message
        self.cv_candidate = cv_candidate
        
        # self.static_report = static_report
        # self.llm_report = llm_report
        
        self.patch = patch
        # self.method = method
        
    def infer_key_variables(self):
        # methods = "\n\n".join(self.methods_list)
        # patches = "\n\n".join(self.patches)
        system_prompt = system_cv_prompt
        
        cv_dict = dict()
        for each_CV in self.cv_candidate:
            # for key1 in self.cv_candidate.keys():
            #     value1 = self.cv_candidate[key1]
            for key1 in each_CV.keys():
                value1 = each_CV[key1]
                for sub_value in value1:
                    for key2 in sub_value.keys():
                        value2 = sub_value[key2]
                        for value3 in value2:
                            if value3 == "":
                                continue
                            for key3 in value3.keys():
                                # for v in value3[key3]:
                                if key3 in cv_dict:
                                    cv_set = cv_dict[key3]
                                else:
                                    cv_set = set()
                                for value3 in value3[key3]:
                                    cv_set.add(value3)
                                cv_dict[key3] = cv_set
        cv_all_list = []
        for key in cv_dict.keys():
            cv_set = cv_dict[key]
            cv_list = []
            for cv in cv_set:
                cv_list.append(cv)
                
            cv_str = ','.join(cv_list)
            cv_str = "[" + cv_str + "]"
            cv_str = key + ": " + cv_str
            cv_all_list.append(cv_str)
            
            
        cv_str = ",".join(cv_all_list)
        # detection_report = (self.static_report + "\\n" + self.llm_report)
        schema = """
        {
            "critical_variables": [
                {
                "variable_name": "",
                "location": ,
                "file": "before/after change",
                "explanation": ""
                },
                ...
            ]
        }
        """
          
        user_prompt = user_cv_prompt_nocontent.format(patches = self.patch, key_variables = cv_str,
                                                message = self.message,
                                                schema = schema)
        
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}]
        
        response = do_request(messages)
        response = response.strip("```json\n").strip("```")
        # start_index = response.find('{')
        # end_index = response.rfind('}')
        
        # response_json_str = response[start_index:end_index + 1]
        
        # 使用正则表达式匹配 JSON 内容
        match = re.search(r'({.*})', response, re.DOTALL)
        if match:
            response_json_str = match.group(1)
        else:
            response_json_str = "{}"  # 默认为空 JSON 对象
        try:
            response_json = json.loads(response_json_str)
        except json.JSONDecodeError:
            print("警告：无法解析 JSON 内容")
            print("原始内容：", response)
            response_json = {}
            
        return response_json, cv_dict
