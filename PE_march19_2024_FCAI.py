import json
import pandas as pd
import subprocess
import os
import csv
import time

def task_selection(df, target_columns, th_val=30):
    task_suggestion_dict = {}
    for col in target_columns:
        unique_num = df[col][df[col].notna()].nunique()
        data = df[col][df[col].notna()]
        ratio = unique_num / data.count()
        
        if df[col].dtypes == "object":
            task_params = "classification"
        else:
            if ratio < 0.05 and unique_num < th_val:
                task_params = "classification"
            else:
                task_params = "regression"
        task_suggestion_dict[col] = task_params

    count_classification = list(task_suggestion_dict.values()).count("classification")
    count_regression = list(task_suggestion_dict.values()).count("regression")
    
    if count_classification > count_regression:
        task_suggestion = "classification"
    else:
        task_suggestion = "regression"
    return task_suggestion
    
def column_selection(df, input_column_names, col_map_invalid_chars):
    input_column_names_str = ', '.join(input_column_names)
    endpoint = os.environ['FUJITSU_CHATAI_ENDPOINT']
    apikey = os.environ['FUJITSU_CHATAI_KEY']
    n=10 
    prompt = "You are the best Machine Learning engineer.\n\n"+"You are provided a tabular dataset with the following columns:\n"+input_column_names_str+"\n\nSelect top-"+str(n)+" columns from the above list which can be used as a target column if you want to create a machine learning model over this dataset. Indicate the confidence level for each response. Provide answer into JSON format with column name as a key and confidence as a value. Sort JSON response by the confidence value. Only provide JSON response without indentation. Do not include long explanation.\n\n"+"Commonly used target column names are \"target\", \"y\", \"class\", \"type\", \"label\" etc. Pay special attention to datasets if it contains such columns. ID (Identifier) columns are not generally used as taget columns. For example \"id\", \"job_id\", \"req_id\", \"roll number\", \"employee id\", \"#\", \"qid\" are not useful as target variables."
    
    curl_command = ['curl', '-s', '-X', 'POST', endpoint, '-H', 'Content-Type: application/json', '-H', 'api-key: '+apikey, '-d', json.dumps({"messages": [{"role":"user", "content": prompt}], "max_tokens": 300, "temperature": 0})]
    curl_execution = subprocess.run(curl_command, capture_output=True, text=True)
    curl_response = {}

    try:
        curl_execution_result = json.loads(curl_execution.stdout)['choices'][0]['message']['content']
        curl_response = json.loads(curl_execution_result)
        column_inverse_map = {col_map_invalid_chars[r]:v for r,v in curl_response.items() if r in col_map_invalid_chars.keys()}
        column_inverse_map_keys = [i for i in column_inverse_map.keys()]
        recommended_column_names = [column_inverse_map_keys[j] for j in range(0,len(column_inverse_map_keys))]+ [l for l in input_column_names if l not in list(curl_response.keys())]
        top1_target_recommendation = [recommended_column_names[0]]
        task_recommendation = task_selection(df, top1_target_recommendation)
        return recommended_column_names, top1_target_recommendation, task_recommendation
    except:
        err = json.loads(curl_execution.stdout)
        recommended_column_names = input_column_names
        top1_target_recommendation = ["Parse error"]
        task_recommendation = ""
        return recommended_column_names, top1_target_recommendation, task_recommendation

def column_selection_task_selection(csv_dataset): 
    colnames = list(csv_dataset.columns)
    tasklist = ["","Classification", "Regression"]

    if len(colnames)>10000:
        print("inside col len exceed block")
        recommended_target_columns = [">10k cols",">10k cols",">10k cols"]
        recommended_ml_task = ""
        return recommended_target_columns, recommended_ml_task
    else:
        input_column_names = []
        colmap = {}
        for c in colnames:
            if not c.lower().startswith('unnamed'):
                res = c.strip()
                colmap[res] = c
                input_column_names.append(res)
            
        recommended_column_names, top1_target_recommendation, stind = column_selection(csv_dataset, input_column_names, colmap)
        recommended_target_columns = recommended_column_names
        recommended_ml_task = stind
        return recommended_target_columns, recommended_ml_task

def gt(data,inpfile):
    for i in data:
        if i['dataset_path']==inpfile:
            return [i['task'], i['target_feature'][0], len(i['target_feature'])>1]
    
def main():
    dataset_path = './datasets/dataframes'
    count = 0
    f = open('GT_target_task.json')
    jsondata = json.load(f)
    with open('results_march19_FCAI.csv', 'a+') as csvfile:
        csvwriter = csv.writer(csvfile)
        for filename in os.listdir(dataset_path):
            tgt, mltsk = [], ""
            if filename.endswith('.csv'):
                filesize = os.path.getsize(dataset_path+'/'+filename)
                print(count, filename, filesize)
                if filesize <= 1073741824:
                    uploadfile = pd.read_csv(dataset_path+'/'+filename)
                    gtvalues = gt(jsondata,filename)
                    tgt, mltsk = column_selection_task_selection(uploadfile)
                    count+=1
                    # column headers- dataset name, predicted target col (top1), GT target col (top1), correct target column (top1), predicted target col (top3), correct target column (top3), predicted task type, GT task type, correct task type, multiple GT target cols  
                    csvwriter.writerow([filename, tgt[0], gtvalues[1], str(tgt[0]==gtvalues[1]), tgt[0:3], str(gtvalues[1] in tgt[0:3]), mltsk, gtvalues[0], str(mltsk==gtvalues[0]), gtvalues[2]])  
                    time.sleep(10)
                else:
                    csvwriter.writerow([filename, ">1G size", ">1G size", ">1G size", ">1G size", ">1G size", ">1G size", ">1G size", ">1G size", ">1G size"]) 

            print("#########")    
    csvfile.close()   
    f.close()                          
    print(count)

main()
