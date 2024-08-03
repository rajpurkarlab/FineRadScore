import pandas as pd
import re
import random

random.seed(197)

# paths to the refisco-v0 and refisco-v1 datasets
REFISCO_v0_PATH = f"datasets/refisco-v0.csv"
REFISCO_v1_PATH = f"datasets/refisco-v1.csv"

# preprocess refisco-v0 to add data_id column
def preprocess_refisco_v0(OUT_FILE):
    refisco_v0 = pd.read_csv(REFISCO_v0_PATH)

    current_data_id = -1
    data_ids = []

    prev_id = None

    for row_index, id in enumerate(refisco_v0["id"]):
        if not prev_id or prev_id != id:
            prev_id = id
            current_data_id += 1
        
        data_ids.append(current_data_id)

    refisco_v0["data_id"] = data_ids
    refisco_v0.to_csv(OUT_FILE)


# preprocess refisco-v1 to add data_id column
def preprocess_refisco_v1(OUT_FILE):
    radiologist_annotations = pd.read_csv(f"datasets/refisco-v1.csv")
    new_data_ids = []
    for data_id, report_type in zip(radiologist_annotations["data_id"], radiologist_annotations["report_type"]):
        new_data_ids.append(str(data_id) + "-" + report_type)
        
    radiologist_annotations["data_id"] = new_data_ids
    radiologist_annotations.to_csv(OUT_FILE)


# consolidate the predictions and ground truths into report format from refisco-v0
def consolidate_pred_gt_v0():
    next_id = 0
    data_df = pd.read_csv(REFISCO_v0_PATH)

    data_ids = []
    study_ids = []
    preds = []
    ground_truths = []
    sources = []
    annotators = []
    pred = None
    ground_truth = None
    prev_id = None
    sentence_id = 0

    for i, study_id in enumerate(data_df["id"]):
        pred_line = data_df["impression_original"][i]
        correction_line = data_df["impression_edited"][i]

        source = data_df["source"][i]
        annotator = data_df["annotator"][i]
        # start a new entry
        if not prev_id or prev_id != study_id:
            # add previous entry
            if pred and ground_truth:
                preds.append(pred.strip())
                ground_truths.append(ground_truth.strip())
            
            pred = ""
            ground_truth = ""
            sentence_id = 0

            study_ids.append(study_id)
            prev_id = study_id

            data_ids.append(next_id)
            next_id += 1

            sources.append(source)
            annotators.append(annotator)

        # apply corrections below
                
        # fix prediction lines
        if pd.isna(pred_line):
            pred_line = ""
            if pd.isna(correction_line):
                correction_line = ""
        
        # fix corrected prediction lines
        if '[delete]' in correction_line:
            correction_line = ""
        elif correction_line == '[no edit]':
            correction_line = pred_line
        
        if pred_line != "":
            pred += f"[{sentence_id}] " + pred_line.strip() + " "
            sentence_id += 1
        
        if correction_line != "":
            ground_truth += correction_line.strip() + " "
    
    # add last remaining entry
    preds.append(pred.strip())
    ground_truths.append(ground_truth.strip())

    consolidated_df = pd.DataFrame({"data_id": data_ids, "study_id": study_ids, "pred": preds, "annotator_ground_truth": ground_truths, "source": sources, "annotator": annotators})
    return consolidated_df


# consolidate the predictions and ground truths into report format from refisco-v1
def consolidate_pred_gt_v1():
    data_df = pd.read_csv(REFISCO_v1_PATH)

    data_ids = []
    subject_ids = []
    study_ids = []
    report_types = []
    preds = []
    ground_truths = []
    annotators = []

    pred = None
    ground_truth = None
    prev_report_text = None
    sentence_id = 0

    for i, data_id in enumerate(data_df["data_id"]):
        pred_line = data_df["original_line"][i]
        correction_line = data_df["corrected_line"][i]

        subject_id = data_df["subject_id"][i]
        study_id = data_df["study_id"][i]
        report_type = data_df["report_type"][i]
        annotator = data_df["annotator_id"][i]

        # start a new entry
        if prev_report_text != data_df["report_text"][i]:
            # add previous entry
            if pred and ground_truth:
                preds.append(pred.strip())
                ground_truths.append(ground_truth.strip())
            
            pred = ""
            ground_truth = ""
            sentence_id = 0

            prev_report_text = data_df["report_text"][i]

            data_ids.append(str(data_id)+"-"+report_type)
            subject_ids.append(subject_id)
            study_ids.append(study_id)
            report_types.append(report_type)
            annotators.append(annotator)

        # apply corrections below
                
        # fix prediction lines
        if pd.isna(pred_line):
            pred_line = ""
            if pd.isna(correction_line):
                correction_line = ""
        
        # fix corrected prediction lines
        if pd.isna(correction_line):
            correction_line = pred_line
        elif '[delete]' in correction_line:
            correction_line = ""
        
        if pred_line != "":
            pred += f"[{sentence_id}] " + pred_line.strip() + " "
            sentence_id += 1
        
        if correction_line != "":
            ground_truth += correction_line.strip() + " "
    
    # add last remaining entry
    preds.append(pred.strip())
    ground_truths.append(ground_truth.strip())

    consolidated_df = pd.DataFrame({"data_id": data_ids, "subject_id": subject_ids, "study_id": study_ids, "report_type": report_types, "pred": preds, "annotator_ground_truth": ground_truths, "annotator": annotators})
    return consolidated_df


# add shuffling of sentences in ground truth report
def consolidate_pred_gt_shuffled(data_df, OUT_FILE):
    shuffled_ground_truths = []
    for ground_truth in data_df['annotator_ground_truth']:
        sentences = re.split(r'(?<!\d)\.(?!\d|$) ', ground_truth)
        if '' in sentences:
            sentences.remove('')
        if ' ' in sentences:
            sentences.remove(' ')
        
        random.shuffle(sentences)

        shuffled_ground_truth = '. '.join(sentences)
        # some postprocessing
        shuffled_ground_truth = shuffled_ground_truth.replace('..', '.')
        while shuffled_ground_truth[-1] == ' ':
            shuffled_ground_truth = shuffled_ground_truth[:-1]
        
        if shuffled_ground_truth[-1] != '.':
            shuffled_ground_truth += '.'
        
        shuffled_ground_truths.append(shuffled_ground_truth)
    
    data_df['shuffled_ground_truth'] = shuffled_ground_truths
    data_df.to_csv(OUT_FILE)


def main():
    print("Starting preprocessing...")
    preprocess_refisco_v0("datasets/refisco-v0_preprocessed.csv")
    preprocess_refisco_v1("datasets/refisco-v1_preprocessed.csv")
    refisco_v0 = consolidate_pred_gt_v0()
    refisco_v1 = consolidate_pred_gt_v1()
    consolidate_pred_gt_shuffled(refisco_v0, "datasets/refisco-v0-input.csv")
    consolidate_pred_gt_shuffled(refisco_v1, "datasets/refisco-v1-input.csv")
    print("Preprocessing complete.")

if __name__=="__main__":
    main()