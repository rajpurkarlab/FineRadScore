import anthropic
import json
import time
import os

from few_shot_prompt import *

# estimate cost of given input and output using claude-3
def estimate_claude3_cost(prompt_tokens, completion_tokens, model="claude-3-sonnet-20240229"):
    if model == "claude-3-sonnet-20240229":
        input_cost = 3
        output_cost = 15
    elif model == "claude-3-opus-20240229":
        input_cost = 15
        output_cost = 75
    else:
        raise "model not recognized"
    
    return (input_cost*prompt_tokens/1000000 + output_cost*completion_tokens/1000000)


# FineRadScore few-shot prompt for claude-3
def generate_claude3_response(pred_target, gt_target):
    client = anthropic.Anthropic(
        api_key=os.environ.get('ANTHROPIC_API_KEY')
    )
    try:
        message = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1024,
            messages=[
                {"role": "user", "content": PROMPT_LONG + """Generated Text: [0] ___M with trauma, evaluate for injuries, pneumothorax. [1] Left lower lung platelike atelectasis. [2] Cardiomegaly. \n \n 
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
    except Exception as e:
        # sleep to avoid rate limiting
        time.sleep(1.2)
        print(e)
        return {"Failed": None}, 0

    try:
        result = json.loads(message.content[0].text)
    except:
        result = {"Failed": None}

    input_tokens = message.usage.input_tokens
    output_tokens = message.usage.output_tokens
    cost = estimate_claude3_cost(input_tokens, output_tokens)

    return result, cost


# FineRadScore zero-shot prompt for claude-3
def generate_claude3_zeroshot_response(pred_target, gt_target):
    client = anthropic.Anthropic(
        api_key=os.environ.get('ANTHROPIC_API_KEY')
    )
    try:
        message = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1024,
            messages=[
                {"role": "user", "content": PROMPT_LONG + JSON_formatting + f"Generated Text: {pred_target} \n \n Ground Truth Text: {gt_target}"}
            ],
        )
    except Exception as e:
        # sleep to avoid rate limiting
        time.sleep(1.2)
        print(e)
        return {"Failed": None}, 0

    try:
        result = json.loads(message.content[0].text)
    except:
        result = {"Failed": None}

    input_tokens = message.usage.input_tokens
    output_tokens = message.usage.output_tokens
    cost = estimate_claude3_cost(input_tokens, output_tokens)

    return result, cost

