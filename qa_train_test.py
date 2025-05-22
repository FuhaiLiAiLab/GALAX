import json
import pandas as pd

def read_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def separate_data(data):
    # Get the training and testing list
    dti_crispr_tr_df = pd.read_csv('./QA_Data/train_dti_crispr_rna_samples.csv')
    dti_crispr_te_df = pd.read_csv('./QA_Data/test_dti_crispr_rna_samples.csv')
    dti_crispr_tr_list = dti_crispr_tr_df['Sample'].tolist()
    dti_crispr_te_list = dti_crispr_te_df['Sample'].tolist()
    # Separate the data into two lists based on tr/te list with (for sample_id, sample_info in data.items())
    tr_data = {}
    te_data = {}
    for sample_id, sample_info in data.items():
        if sample_id in dti_crispr_tr_list:
            tr_data[sample_id] = sample_info
        elif sample_id in dti_crispr_te_list:
            te_data[sample_id] = sample_info
    # Check if the separated data is correct
    tr_data_count = len(tr_data)
    te_data_count = len(te_data)
    print(f"Training data count: {tr_data_count}")
    print(f"Testing data count: {te_data_count}")
    # Check if the training and testing data are mutually exclusive
    if set(tr_data.keys()).intersection(te_data.keys()):
        print("Error: Training and testing data are not mutually exclusive.")
    else:
        print("Training and testing data are mutually exclusive.")
    return tr_data, te_data

def save_to_json(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)
    print(f"Data saved to {file_path}")

def main():
    # Read the JSON file
    data = read_json('./QA_Data/multi_sample_qa_info_k100_bm100.json')
    
    # Separate the data into training and testing sets
    tr_data, te_data = separate_data(data)
    
    # Save the separated data to new JSON files
    save_to_json(tr_data, './QA_Data/multi_sample_qa_info_k100_bm100_tr.json')
    save_to_json(te_data, './QA_Data/multi_sample_qa_info_k100_bm100_te.json')

if __name__ == "__main__":
    main()
    