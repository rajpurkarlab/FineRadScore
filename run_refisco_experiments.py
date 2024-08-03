import pandas as pd
import json
from tqdm import tqdm
import sys

from gpt4_generations import generate_gpt4_response, generate_gpt4_response_zeroshot
from claude3_generations import generate_claude3_response, generate_claude3_zeroshot_response

def generate_fineradscore_corrections(pred_gt_df, GROUND_TRUTH_COLUMN, OUT_FILE, model, zeroshot, PRED_COLUMN=None, max_retries=5):
    print("Using model:", model)
    print("Using fewshot:", not zeroshot)

    if not PRED_COLUMN:
        PRED_COLUMN = "pred"
    
    data_ids = []
    pred_targets = []
    gt_targets = []
    gpt4_responses = []

    total_cost = 0

    for data_id in tqdm(pred_gt_df["data_id"].unique()):
        data_ids.append(data_id)

        # get pred and ground truth targets
        pred_target = pred_gt_df[pred_gt_df["data_id"] == data_id][PRED_COLUMN].values[0]
        gt_target = pred_gt_df[pred_gt_df["data_id"] == data_id][GROUND_TRUTH_COLUMN].values[0]

        pred_targets.append(pred_target)
        gt_targets.append(gt_target)

        done = False
        retry_count = 0
        while not done:
            done = True

            try:
                if model == "gpt4":
                    if zeroshot:
                        result, cost = generate_gpt4_response_zeroshot(pred_target, gt_target)
                    else:
                        result, cost = generate_gpt4_response(pred_target, gt_target)
                else:
                    if zeroshot:
                        result, cost = generate_claude3_zeroshot_response(pred_target, gt_target)
                    else:
                        result, cost = generate_claude3_response(pred_target, gt_target)
            except:
                done = False
                continue

            total_cost += cost
            
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
        
        gpt4_responses.append(json.dumps(result))
        print(f"Total cost: {total_cost}")


    results = {"data_id": data_ids, "pred": pred_targets, "ground_truth": gt_targets, "claude3_raw_response": gpt4_responses}
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT_FILE)


def run_refisco_experiment(version, setting, model):
    OUT_FILE = f"{model}_results/refisco_{version}/{setting}/results_raw.csv"

    if setting == "paraphrased":
        pred_gt_df = pd.read_csv(f"datasets/refisco-{version}-paraphrased.csv")
    else:
        pred_gt_df = pd.read_csv(f"datasets/refisco-{version}-input.csv")
    
    if setting == "shuffled":
        GROUND_TRUTH_COLUMN = "shuffled_ground_truth"
    else:
        GROUND_TRUTH_COLUMN = "annotator_ground_truth"

    generate_fineradscore_corrections(pred_gt_df, GROUND_TRUTH_COLUMN, OUT_FILE, model, zeroshot=setting=="zeroshot")


def main():
    # correct usage: python run_refisco_experiments.py <version> <setting> <model>
    if len(sys.argv) != 4:
        print("Invalid number of arguments")
        return
    
    version = sys.argv[1]
    setting = sys.argv[2]
    model = sys.argv[3]

    assert version in ["v0", "v1"]
    assert setting in ["zeroshot", "original", "shuffled", "paraphrased"]
    assert model in ["gpt4", "claude3"]

    run_refisco_experiment(version, setting, model)


if __name__=="__main__": 
    main()

