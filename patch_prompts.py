system_analyze_prompt = """
ou are a software defect analysis expert. Analyze the provided commit description and patch to determine if this is a defect-fixing commit. A commit qualifies as a bug fix only if:
Defect Resolution:
    - The patch directly addresses a demonstrable software defect (e.g., crashes, incorrect behavior, security flaws).
    - Pre-patch code contains a clear issue (e.g., logic errors, invalid memory access, input validation gaps).
    - Post-patch code eliminates the defect through code-level remediation (e.g., fixes logic, adds validation, removes unsafe patterns).

Exclusions (Not bug fixes):
    - Code style/formatting changes.
    - Performance improvements unrelated to defects.
    - New features/enhancements.
    - Documentation/test updates.
    - Preventive measures for hypothetical issues.
"""

user_analyze_prompt = '''
I will provide you with the following information:
    1. Submission description:
        {description}
    2. Submit patches and Function where the patches are located: 
        {patches}
Please determine whether the patch has fixed existing bug based on the above information. Please note that the definition of bug repair, functional upgrades and improvements, and security protection upgrades are not considered as bug repairs.    
If the functions information is empty, please make a judgment based on the description and content of the patch.If the description of a commit contains keywords such as fix, bug, etc., these commits tend to be bug fixes.
'''

user_analyze_nomethod_prompt = '''
I will provide you with the following information:
    1. Submission description:
        {description}
    2. Submit patches: 
        {patches}
Please determine whether the patch has fixed existing bug based on the above information. Please note that the definition of bug repair, functional upgrades and improvements, and security protection upgrades are not considered as bug repairs.    
If the functions information is empty, please make a judgment based on the description and content of the patch.If the description of a commit contains keywords such as fix, bug, etc., these commits tend to be bug fixes.
'''


json_prompt = """
Please output in JSON format. The content of the target JSON file is as follows:
{
    "answer": "whether the commit is a bug fix commit",
    "analyze": "detailed analysis process and reasons"
}

Example:
{
    "answer": "yes",
    "analyze": "xxxx."
}
"""
