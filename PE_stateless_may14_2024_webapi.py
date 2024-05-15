import json
import subprocess
import os

# ======= Main funtion call =======
# column_selection_task_selection(csv_dataset)

# ======= Output =======
# if entire workflow is successful, returns a list of strings with four items
#   (i) top1 predicted target column 
#   (ii) explanation of most probable target columns (upto 10)
#   (iii) predicted ML task 
#   (iv) explanation of ML task 
# else returns error (four possible types) with appropriate message in JSON format i.e. {'Error': 'Error description'}
#   (i) {"Error": "LLM endpoint and key pairs could not be validated."}
#   (ii) {"Error": "Parsing of LLM response was not successful."}
#   (iii) {"Error": "The dataset contains more than 10000 columns so target column has to be specified manually."}
#   (iv) {"Error": "Environment variables for AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_API_KEY are not set up."}

# function to predict ML task type
def task_selection(df, target_columns, th_val=30):
    # temporary variable to store intermediate results
    task_suggestion_dict = {}
    explanation_tasktype = ""

    for col in target_columns:
        # for each target column, calculate the ratio of unique values present in the column with respect to all not empty values
        unique_num = df[col][df[col].notna()].nunique()
        data = df[col][df[col].notna()]
        ratio = unique_num / data.count()
        
        # case when target columns is of string/categorical/object type (i.e. classification)
        if df[col].dtypes == "object":
            task_params = "Classification"
            explanation_tasktype += "The target column '" + col + "' is of categorical/boolean/string data type. " 
        else:
            # case when target column is of numeric type with limited number of possible values (i.e. classification)
            if ratio < 0.05 and unique_num < th_val:
                task_params = "Classification"
                explanation_tasktype += "The target column '" + col + "' is of numerical data type with "+str(unique_num)+" unique values. " 
            # case when target columns is of numeric type and wide spread of possible values (i.e. regression)
            else:
                task_params = "Regression"
                explanation_tasktype += "The selected target column '" + col + "' is of numerical data type with "+str(unique_num)+" unique values. "
        task_suggestion_dict[col] = task_params

    count_classification = list(task_suggestion_dict.values()).count("Classification")
    count_regression = list(task_suggestion_dict.values()).count("Regression")
    
    # if more number of target columns are suited for classification task
    if count_classification > count_regression:
        task_suggestion = "Classification"
        explanation_tasktype += "A majority of the target columns are suitable for the Classification task."
    # if more number of target columns are suited for regression task
    else:
        task_suggestion = "Regression"
        explanation_tasktype += "A majority of the target columns are suitable for the Regression task."
    # return task type with appropriate explanation
    return task_suggestion, explanation_tasktype

# function to predict target column
def column_selection(df, input_column_names, col_map_invalid_chars):
    
    # input_column_names as a string (which will be a dynamic argument within the prompt conditioned on the dataset uploaded)
    input_column_names_str = ', '.join(input_column_names)
    
    # (i) endpoint (ii) apikey (iii) number of recommendations desired (iv) prompt template for target column prediction using Azure OpenAI 
    endpoint = os.environ['AZURE_OPENAI_ENDPOINT']
    apikey = os.environ['AZURE_OPENAI_API_KEY']        
    n=10
    prompt = "You are the best Machine Learning engineer.\n\n"+"You are provided a tabular dataset with the following columns:\n"+input_column_names_str+"\n\nSelect top-"+str(n)+" columns from the above list which can be used as a target column if you want to create a machine learning model over this dataset. Indicate the confidence level for each response. Provide answer into JSON format with column name as a key and confidence as a value. Sort JSON response by the confidence value. Only provide JSON response without indentation. Do not include long explanation.\n\n"+"Commonly used target column names are \"target\", \"y\", \"class\", \"type\", \"label\" etc. Pay special attention to datasets if it contains such columns. ID (Identifier) columns are not generally used as taget columns. For example \"id\", \"job_id\", \"req_id\", \"roll number\", \"employee id\", \"#\", \"qid\" are not useful as target variables."
    
    # curl command formation and execution for making Azure OpenAI request
    curl_command = ['curl', '-s', '-i', '-X', 'POST', endpoint, '-H', 'Content-Type: application/json', '-H', 'api-key: '+apikey, '-d', json.dumps({"messages": [{"role":"user", "content": prompt}], "max_tokens": 300, "temperature": 0})]
    curl_execution = subprocess.run(curl_command, capture_output=True, text=True)
    
    # if response code is other than 200 (success), throw error
    if not curl_execution.stdout.startswith('HTTP/2 200'):
        return {"Error": "LLM endpoint and key pairs could not be validated."} 
    # if response code is 200 (success), extract necessary information from response to recommend target column
    else:
        # temporary variable to store curl response
        curl_response = {}

        try:
            # parse response from Azure OpenAI request
            curl_execution_result = json.loads(curl_execution.stdout.split("\n")[-2])['choices'][0]['message']['content']
            curl_response = json.loads(curl_execution_result)
            
            # column names are post-processed to avoid any errors in LLM calls, so now do inverse mapping to get original column names back
            column_inverse_map = {col_map_invalid_chars[r]:v for r,v in curl_response.items() if r in col_map_invalid_chars.keys()}
            column_inverse_map_keys = [i for i in column_inverse_map.keys()]
            
            # rank columns based on their probability of being target columns
            recommended_column_names = [column_inverse_map_keys[j] for j in range(0,len(column_inverse_map_keys))]+ [l for l in input_column_names if l not in list(curl_response.keys())]
            top1_target_recommendation = [recommended_column_names[0]]

            # call task_selection with dataframe and top1_target_recommendation
            task_recommendation, expl_tasktype = task_selection(df, top1_target_recommendation)

            # generate appropriate explanation for the target column
            expl_targetcol = "Top recommendations (by Azure OpenAI API) for target colums are: ["+', '.join(column_inverse_map_keys)+"]"
            
            return [top1_target_recommendation[0], expl_targetcol, task_recommendation, expl_tasktype]
        except:
            # if LLM response cannot be parsed properly, throw error
            return {"Error": "Parsing of LLM response was not successful."}

# main function which will call column_selection and task_selection
def column_selection_task_selection(csv_dataset): 
    
    colnames = list(csv_dataset.columns)
    
    # if there are more than 10000 columns in the dataset, throw error
    if len(colnames)>10000:
        return {"Error": "The dataset contains more than 10000 columns so target column has to be specified manually."}
    # if LLM endpoint and api key are not configured properly, throw error
    elif 'AZURE_OPENAI_ENDPOINT' not in os.environ or 'AZURE_OPENAI_API_KEY' not in os.environ:    
        return {"Error": "Environment variables for AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_API_KEY are not set up."}
    # if LLM endpoint and api key are configured properly, call necessary functions to predict target column using LLM
    elif 'AZURE_OPENAI_ENDPOINT' in os.environ and 'AZURE_OPENAI_API_KEY' in os.environ:
        
        # initialize temporary data structures and post-process column names to avoid any errors in LLM calls (whitespace removal, esacpe characters etc.);
        input_column_names = []
        colmap = {}
        for c in colnames:
            if not c.lower().startswith('unnamed'):
                res = c.strip()
                colmap[res] = c
                input_column_names.append(res)
            
        return column_selection(csv_dataset, input_column_names, colmap)
