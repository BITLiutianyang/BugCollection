import javalang

class JavaMethodExtractor:
    def __init__(self, file_content):
        self.file_content = file_content
        self.java_code = ""
        self.method_bodies = {}
        self.method_bodies_context = {}
        self.func_linenumber = {}
        self.ast_tree = None
        self.line_offsets = []
        self.call_graph = {}
        self.reversed_call_graph = {}
        
        self.method_lines = {}
        self.lines_list = []
        
        self.method_node = {}
        
        self.method_params = {}
        
        self.method_params_callee = {} # 调用位置所在的参数
        
        self.callee_lines = {}
        
        self.global_context = {
            'imports': [],
            'classes': {}
        }
        
        self.generate_method()
        self.generate_call_graph()
    def load_file(self):
        """加载并解析 Java 文件"""
        self.lines_list = []
        try:
            # with open(self.file_path, "r", encoding="utf-8") as f:
            #     self.java_code = f.read()
            # with open(self.file_path, "r", encoding="utf-8") as f:
            self.java_code = self.file_content
            
            lines = self.java_code.split('\n')
            index = 1
            for line in lines:
                self.lines_list.append(line)
                index += 1

            self.ast_tree = javalang.parse.parse(self.java_code)
        except Exception as e:
            print(f"加载并解析 Java 文件时出错: {e}")
            self.ast_tree = None  
            
    def find_matching_brace(code, start_idx):
        """查找配对的大括号"""
        stack = []
        lines = code.split('\n')
        line_start_indices = [0]
        for line in lines:
            line_start_indices.append(line_start_indices[-1] + len(line) + 1)

        for idx in range(start_idx, len(code)):
            if code[idx] == '{':
                stack.append(idx)
            elif code[idx] == '}':
                if stack:
                    stack.pop()
                    if not stack:
                        # 计算行号和列号
                        line = 0
                        while line_start_indices[line] <= idx:
                            line += 1
                        return idx, line
        return None, None
    
    def pre_process_line_offsets(self):
        """预处理行号到字符索引的映射"""
        if not self.java_code:
            return
        lines = self.java_code.split('\n')
        self.line_offsets = [0]
        for line in lines:
            self.line_offsets.append(self.line_offsets[-1] + len(line) + 1)  # +1 为换行符
    
    
    def extract_method_bodies(self):
        """提取方法体"""
        if not self.ast_tree:
            return
        
        self.pre_process_line_offsets()
        
        for path, node in self.ast_tree.filter(javalang.tree.Import):
            self.global_context['imports'].append(node)
        
        for path, node in self.ast_tree.filter(javalang.tree.ClassDeclaration):
            class_name = node.name
            self.global_context['classes'][class_name] = {
                'class_members': [],
                'methods': {}
            }
            for member in node.body:
                if isinstance(member, javalang.tree.FieldDeclaration):
                    self.global_context['classes'][class_name]['class_members'].append(member)

                elif isinstance(member, javalang.tree.MethodDeclaration):
                    method_name = member.name
                    self.global_context['classes'][class_name]['methods'][method_name] = {
                        'parameters': member.parameters,
                        'return_type': member.return_type
                    }
                elif isinstance(member, javalang.tree.ConstructorDeclaration):
                    method_name = member.name
                    self.global_context['classes'][class_name]['methods'][method_name] = {
                        'parameters': member.parameters,
                        'return_type': None
                    }
                    

    def generate_method_bodies(self):
        """生成方法体"""
        if not self.java_code:
            self.load_file()
        
        self.pre_process_line_offsets()
        
        
        method_declarations = self.ast_tree.filter(javalang.tree.MethodDeclaration)
        constuction_declarations = self.ast_tree.filter(javalang.tree.ConstructorDeclaration)
        all_declarations = list(constuction_declarations) + list(method_declarations)
        for path, node in all_declarations:
            method_name = node.name
            start_line = node.position.line
            self.method_params[(method_name, start_line)] = node.parameters
            if not node.body:
                continue  # 方法体为空，跳过
            
            # 获取方法体的起始位置（左大括号的位置）
            
            # start_line = node.body[0]  # 方法体的第一个块语句
            # start_line = block.position.line
            start_col = node.position.column
            start_idx = self.line_offsets[start_line - 1] + (start_col - 1)
            
            # 查找配对的右大括号
            brace_end, end_line = JavaMethodExtractor.find_matching_brace(self.java_code, start_idx)
            if brace_end is None:
                continue  # 括号不匹配，跳过
            
            end_idx = brace_end + 1  # 包含右大括号
            method_body = self.java_code[start_idx:end_idx].strip()
            method_body_context = "".join(self.lines_list[start_line: end_line- 1])
            
            # 使用方法名和起始行号作为键
            self.method_bodies[(method_name, start_line)] = method_body
            self.method_bodies_context[(method_name, start_line)] = method_body_context
            
            self.method_lines[(method_name, start_line)] = [start_line, end_line]
    
    def build_call_graph(self):
        """构建调用图及相关信息"""
        if not self.ast_tree:
            self.load_file()
        
        if self.ast_tree is None:
            return
        
        # 收集所有方法的行号
        method_declarations = self.ast_tree.filter(javalang.tree.MethodDeclaration)
        constuction_declarations = self.ast_tree.filter(javalang.tree.ConstructorDeclaration)
        all_declarations = list(method_declarations) + list(constuction_declarations)
        for path, node in all_declarations:
            method_name = node.name
            method_line = node.position.line if node.position else -1

            if method_name not in self.func_linenumber:
                self.func_linenumber[method_name] = []
            if method_line not in self.func_linenumber[method_name]:
                self.func_linenumber[method_name].append(method_line)
                
            if (method_name, method_line) not in self.method_node:
                self.method_node[(method_name, method_line)] = []
            self.method_node[(method_name, method_line)].append(node)

        # 构建调用图
        for path, node in self.ast_tree.filter(javalang.tree.MethodInvocation):
            caller = None
            caller_line = None
            for node_in_path in reversed(path):
                if isinstance(node_in_path, (javalang.tree.MethodDeclaration, javalang.tree.ConstructorDeclaration)):
                    caller = node_in_path.name
                    caller_line = node_in_path.position.line if node_in_path.position else None
                    break
                elif isinstance(node_in_path, javalang.tree.Annotation):
                    break

            method_name = node.member
            method_line = node.position.line if node.position else None
            method_params = node.arguments

            # 确定调用者和被调用方法
            caller_key = (caller or "UNKNOWN", caller_line or -1)
            callee_key = (method_name or "UNKNOWN", method_line or -1)

            # 更新调用图
            if caller_key not in self.call_graph:
                self.call_graph[caller_key] = []
            self.call_graph[caller_key].append(callee_key)
            
            if callee_key not in self.method_params_callee:
                self.method_params_callee[callee_key] = []
            self.method_params_callee[callee_key] = method_params
            
            self.callee_lines[callee_key] = self.lines_list[method_line - 1]
                

            # 更新逆向调用图
            if method_name not in self.reversed_call_graph:
                self.reversed_call_graph[method_name] = []
            self.reversed_call_graph[method_name].append(caller_key)
    
    

            
            
    def generate_call_graph(self):
        """获取函数调用图"""
        
        if not self.call_graph:
            self.build_call_graph()
            
        return self.call_graph
    
    
    def generate_method(self):
        """获取方法体字典"""
        self.load_file()
        if self.ast_tree:
            self.extract_method_bodies()
            self.generate_method_bodies()
        return self.method_bodies
    
    
    def get_call_graph(self):
        """获取调用图"""
        return self.call_graph

    def get_reversed_call_graph(self):
        """获取逆向调用图"""
        return self.reversed_call_graph

    def get_func_linenumber(self):
        """获取方法定义的行号信息"""
        return self.func_linenumber
    
    def get_method_bodies(self):
        return self.method_bodies
    
    def get_method_bodies_context(self):
        return self.method_bodies_context
    
    def get_method_lines(self):
        return self.method_lines
    
    def get_method_node(self):
        return self.method_node
    
    def get_method_params(self):
        return self.method_params
    
    def get_method_params_callee(self):
        return self.method_params_callee
    
    def get_global_context(self):
        return self.global_context
    
    def get_callee_lines(self):
        return self.callee_lines
