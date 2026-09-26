"""MMMU open-answer shape fix, applied uniformly before full evaluation outcomes.

The pinned v0.7.2 helper collapses parsed open answers to one string, although
eval_open expects a list of normalized strings/numbers. Preserve that list for
each generated candidate, matching the official MMMU evaluator contract.
Question rendering, multiple-choice parsing, and aggregation are unchanged.
"""
import ast
from mmmu_utils import (get_multi_choice_info,parse_multi_choice_response,
                        parse_open_response,extract_subset_name)

def mmmu_process_results(doc,results):
    parsed=[]
    for response in results:
        if doc["question_type"]=="multiple-choice":
            info,choices=get_multi_choice_info(ast.literal_eval(doc["options"]))
            prediction=parse_multi_choice_response(response,choices,info)
        else:prediction=parse_open_response(response)
        parsed.append(prediction)
    record={"id":doc["id"],"subdomain":extract_subset_name(doc["id"]),"question_type":doc["question_type"],"answer":doc["answer"],"parsed_pred":parsed}
    return {"mmmu_acc":record,"mmmu_acc_pass_at_k":record,"submission":{doc["id"]:parsed[0]}}
