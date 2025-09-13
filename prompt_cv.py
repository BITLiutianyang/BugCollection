system_cv_prompt = """
As a software defect analysis expert, your task is to identify **critical variables** from a bug-fix commit, those variables that directly trigger defects in before-commit code and repair defects in after-commit code.
These critical variables must satisfy the following two criteria simultaneously:
1. Be explicitly modified/added/deleted in the patch.
2. Be directly related to the root cause of the defect.

**Examples**:
1. **Null Pointer Fix**: 
   - Critical Variable: `object_ptr` (previously uninitialized; initialized or null-checked in the patch).
2. **Boundary Condition Fix**:
   - Critical Variable: `index` (previously incorrect increment logic in loop condition, corrected in the patch).
"""

user_cv_prompt_nocontent = '''
Please analyze the following information and identify **critical variables** step-by-step:
    1. **Patch Changes**: 
        {patches}
    2. **Static Analysis Suspects Variables**:
        {key_variables}
    3. **Commit Message**:
        {message}
---

### **Your Task**:
1. Identify the variables modified/added/deleted explicitly in the patch.

2. Determine the intersection between these variables and those suggested by static analysis.

3. Analyze how each intersecting variable is logically connected to the root cause of the defect.

4. Output critical variables strictly in the following JSON format, do not output anything else.

---

### **Expected JSON Output Schema**:
```json
{schema}
'''
