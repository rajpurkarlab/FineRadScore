import os
import openai
import json
import time

from few_shot_prompt import *

openai.api_type = "azure"
openai.api_version = "2023-05-15" 
openai.api_base = "https://ahuang.openai.azure.com/"
openai.api_key = os.environ.get('OPENAI_API_KEY')

# estimate cost of given input and output using gpt-4
def estimate_gpt4_cost(prompt_tokens, completion_tokens):
    input_cost = 0.03
    output_cost = 0.06
    
    return (input_cost*prompt_tokens/1000 + output_cost*completion_tokens/1000)


# FineRadScore few-shot prompt for gpt-4
def generate_gpt4_response(pred_target, gt_target):
    response = openai.ChatCompletion.create(
        engine="gpt4",
        messages=[
            {"role": "system", "content": PROMPT_LONG},
            {"role": "user", "content": """Generated Text: [0] ___M with trauma, evaluate for injuries, pneumothorax. [1] Left lower lung platelike atelectasis. [2] Cardiomegaly. \n \n 
                                            Ground Truth Text: "Right lower lung consolidation, either pneumonia, aspiration, or possibly pulmonary contusions from recent trauma. No evidence of displaced rib fracture or pneumothorax."""},
            {"role": "assistant", "content": json.dumps(corrections1)},
            {"role": "user", "content": """Generated Text: [0] Three left lung nodules concerning for metastatic disease. [1] Multiple lung nodules \n \n 
                                            Ground Truth Text: "Two left lung nodules concerning for metastatic disease. Left basilar opacity could represent atelectasis or consolidation."""},
            {"role": "assistant", "content": json.dumps(corrections2)},
            {"role": "user", "content": """Generated Text: [0] Stable position of endotracheal tube projects 2.2 cm above the carina. [1] Minimal atelectasis at the right lung base. [2] Moderate cardiomegaly. [3] Pulmonary edema. [4] The presence of a minimal left pleural effusion cannot be excluded. \n \n 
                                            Ground Truth Text: "Endotracheal tube projects approximately 2.2 cm above the carina. Minimal atelectasis at the left and right lung bases. Moderate cardiomegaly. The presence of a minimal left pleural effusion cannot be excluded."""},
            {"role": "assistant", "content": json.dumps(corrections3)},
            {"role": "user", "content": """Generated Text: [0] Endotracheal tube is in standard position. [1] A nasogastric tube is seen coursing into the stomach with tip in the stomach. [2] Heart size is normal. [3] Lungs are clear. [4] No pleural effusion or pneumothorax. [5] Enteric tube tip is in the stomach. \n \n 
                                            Ground Truth Text: "Endotracheal tube is in standard position. Heart size is normal. Lungs are clear. No pleural effusion or pneumothorax."""},
            {"role": "assistant", "content": json.dumps(corrections4)},
            {"role": "user", "content": """Generated Text: [0] The lungs are well expanded. [1] There is no pleural effusion or pneumothorax. [2] The cardiomediastinal and hilar contours are unremarkable. \n \n 
                                            Ground Truth Text: "The lungs are adequately inflated. The contours of the cardiomediastinal and hilar regions appear normal. There are no indications of pleural effusion or pneumothorax."""},
            {"role": "assistant", "content": json.dumps(corrections5)},

            {"role": "user", "content": f"Generated Text: {pred_target} \n \n Ground Truth Text: {gt_target}"}
        ],
    )

    try:
        result = json.loads(response['choices'][0]['message']['content'])
    except:
        result = {"Failed": None}
    
    cost = estimate_gpt4_cost(response["usage"]["prompt_tokens"], response["usage"]["completion_tokens"])

    return result, cost


# FineRadScore zero-shot prompt for gpt-4
def generate_gpt4_response_zeroshot(pred_target, gt_target):
    response = openai.ChatCompletion.create(
        engine="gpt4",
        messages=[
                {"role": "user", "content": PROMPT_LONG + JSON_formatting + f"Generated Text: {pred_target} \n \n Ground Truth Text: {gt_target}"}
        ],
    )

    try:
        result = json.loads(response['choices'][0]['message']['content'])
    except:
        result = {"Failed": None}
    
    cost = estimate_gpt4_cost(response["usage"]["prompt_tokens"], response["usage"]["completion_tokens"])

    return result, cost
