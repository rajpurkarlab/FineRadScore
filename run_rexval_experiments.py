import re
import json
import pandas as pd
from tqdm import tqdm
import sys

from gpt4_generations import generate_gpt4_response, generate_gpt4_response_zeroshot
from claude3_generations import generate_claude3_response, generate_claude3_zeroshot_response

def run_rexval_experiment(model, setting, full=False, max_retries=5):
    print("Using model:", model)
    print("Using fewshot:", setting == "fewshot")

    PRED_COLUMN = "pred_report"
    GROUND_TRUTH_COLUMN = "gt_report"

    if full:
        OUT_FILE = f"opus_results/rexval_full/results_raw_categories.csv"
        pred_gt_df = pd.read_csv(f"datasets/rexval_full.csv")
    else:
        OUT_FILE = f"opus_results/rexval_test/results_raw_categories.csv"
        pred_gt_df = pd.read_csv(f"datasets/ReXVal_test_40.csv")

    for row_index, row in tqdm(pred_gt_df.iterrows(), total=len(pred_gt_df)):
        pred = row[PRED_COLUMN]
        gt = row[GROUND_TRUTH_COLUMN]

        # add sentence ids to pred
        sentences = re.split(r'(?<!\d)\.(?!\d|$) ', pred)
        if '' in sentences:
                sentences.remove('')
        if ' ' in sentences:
            sentences.remove(' ')
        
        for i, sentence in enumerate(sentences):
            # some cleaning
            sentence = sentence.strip()
            sentence = sentence.replace('"', '')

            sentence = sentence + "."
            sentences[i] = sentence.replace("..", ".")
        
        pred = ""
        for sentence_id, sentence in enumerate(sentences):
            # add the sentences with ids for LLM input
            pred += f"[{sentence_id}] " + sentences[sentence_id] + " "
        
        pred = pred.strip()

        done = False
        retry_count = 0
        while not done:
            done = True

            try:
                if model == "gpt4":
                    if setting == "zeroshot":
                        result, cost = generate_gpt4_response_zeroshot(pred, gt)
                    else:
                        result, cost = generate_gpt4_response(pred, gt)
                else:
                    if setting == "zeroshot":
                        result, cost = generate_claude3_zeroshot_response(pred, gt)
                    else:
                        result, cost = generate_claude3_response(pred, gt)
            except:
                done = False
                continue
            
            for sentence_id in result:
                # bad generation: regenerate
                if not sentence_id.isdigit() and sentence_id != "None":
                    done = False
                    print("Error: key is not a sentence id")
                    break

                # bad generation: regenerate
                corrected_line = result[sentence_id]
                if "corrections" not in corrected_line or "clinical severity" not in corrected_line or "comments" not in corrected_line or "error category" not in corrected_line:
                    done = False
                    print("Error: json object not formatted correctly")
                    break
            
            retry_count += 1
            if retry_count > max_retries:
                done = True
                break
        
        pred_gt_df.at[row_index, "claude3_raw_response"] = json.dumps(result)

    pred_gt_df.to_csv(OUT_FILE, index=False)


def main():
    # correct usage: python run_rexval_experiments.py <version> <setting> <model>
    if len(sys.argv) != 4:
        print("Invalid number of arguments")
        return
    
    version = sys.argv[1]
    setting = sys.argv[2]
    model = sys.argv[3]

    assert version in ["test", "full"]
    assert setting in ["zeroshot", "fewshot"]
    assert model in ["gpt4", "claude3"]

    run_rexval_experiment(model, setting, full=version=="full")


if __name__=="__main__": 
    main()
