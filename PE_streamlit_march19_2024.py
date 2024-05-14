import json
import pandas as pd
import subprocess
import streamlit as st
import os

class TargetTaskRecommendation:
    
    def task_selection(_self, df, target_columns, th_val=30):
        """
            Predict ML Task Type
            Parameters:
            - df (pd.DataFrame): The training dataset
            - target_columns (list): List of columns selected as target columns
            - th_val (int, default value 30): Threshold to determine whether classification or regression is more appropriate
            Returns:
            - "Regression"/"Classification" (string)
        """
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
                explanation_tasktype += "The selected target column '" + col + "' is of categorical/boolean/string data type.\n" 
            else:
                # case when target column is of numeric type with limited number of possible values (i.e. classification)
                if ratio < 0.05 and unique_num < th_val:
                    task_params = "Classification"
                    explanation_tasktype += "The selected target column '" + col + "' is of numerical data type with "+str(unique_num)+" unique values.\n" 
                # case when target columns is of numeric type and wide spread of possible values (i.e. regression)
                else:
                    task_params = "Regression"
                    explanation_tasktype += "The selected target column '" + col + "' is of numerical data type with "+str(unique_num)+" unique values.\n"
            task_suggestion_dict[col] = task_params

        count_classification = list(task_suggestion_dict.values()).count("Classification")
        count_regression = list(task_suggestion_dict.values()).count("Regression")
        
        # if more number of target columns are suited for classification task
        if count_classification > count_regression:
            task_suggestion = "Classification"
            explanation_tasktype += "A majority of the target columns are suitable for the Classification task.\n"
        # if more number of target columns are suited for regression task
        else:
            task_suggestion = "Regression"
            explanation_tasktype += "A majority of the target columns are suitable for the Regression task.\n"
        # generate appropriate explanation for the task type
        st.session_state.expltt = explanation_tasktype
        return task_suggestion
    
    # cache column_selection function to prevent unnecessary LLM calls on user intractions
    @st.cache_data(show_spinner=False)
    def column_selection(_self, df, input_column_names, col_map_invalid_chars):
        """
            Recommend top-10 Target Columns using Azure ChatAPI
            Parameters:
            - df (pd.DataFrame): The training dataset
            - input_column_names (list): List of column names (post-processed) from the training dataset
            - col_map_invalid_chars (dict): Column names are post-processed to avoid any errors in LLM calls (whitespace removal, esacpe characters etc.); This dictionary maps original column names and post-processed column names 
            Returns:
            - recommended_column_names (list): List of column names sorted by the probability of being target column recommended by LLM for the given dataset
            - top1_target_recommendation (string): Most probable target column recommended by LLM for the given dataset
            - task_recommendation (string): Whether Recommendation or Classification task is more appropriate for the given dataset
        """
        # input_column_names as a string (which will be a dynamic argument within the prompt conditioned on the dataset uploaded)
        input_column_names_str = ', '.join(input_column_names)
        
        # (i) endpoint (ii) apikey (iii) number of recommendations desired (iv) prompt template for target column prediction using Azure OpenAI 
        endpoint = os.environ['AZURE_OPENAI_ENDPOINT']
        apikey = os.environ['AZURE_OPENAI_API_KEY']
        n=10 
        prompt = "You are the best Machine Learning engineer.\n\n"+"You are provided a tabular dataset with the following columns:\n"+input_column_names_str+"\n\nSelect top-"+str(n)+" columns from the above list which can be used as a target column if you want to create a machine learning model over this dataset. Indicate the confidence level for each response. Provide answer into JSON format with column name as a key and confidence as a value. Sort JSON response by the confidence value. Only provide JSON response without indentation. Do not include long explanation.\n\n"+"Commonly used target column names are \"target\", \"y\", \"class\", \"type\", \"label\" etc. Pay special attention to datasets if it contains such columns. ID (Identifier) columns are not generally used as taget columns. For example \"id\", \"job_id\", \"req_id\", \"roll number\", \"employee id\", \"#\", \"qid\" are not useful as target variables."
        
        # curl command formation and execution for making Azure OpenAI request
        curl_command = ['curl', '-s', '-X', 'POST', endpoint, '-H', 'Content-Type: application/json', '-H', 'api-key: '+apikey, '-d', json.dumps({"messages": [{"role":"user", "content": prompt}], "max_tokens": 300, "temperature": 0})]
        curl_execution = subprocess.run(curl_command, capture_output=True, text=True)
        
        # temporary variable to store curl response
        curl_response = {}

        try:
            # parse response from Azure OpenAI request
            curl_execution_result = json.loads(curl_execution.stdout)['choices'][0]['message']['content']
            curl_response = json.loads(curl_execution_result)
            
            # column names are post-processed to avoid any errors in LLM calls, so now do inverse mapping to get original column names back
            column_inverse_map = {col_map_invalid_chars[r]:v for r,v in curl_response.items()}
            column_inverse_map_keys = [i for i in column_inverse_map.keys()]
            
            # rank columns based on their probability of being target columns
            recommended_column_names = [column_inverse_map_keys[j] for j in range(0,len(column_inverse_map_keys))]+ [l for l in input_column_names if l not in list(curl_response.keys())]
            top1_target_recommendation = [recommended_column_names[0]]

            # call task_selection with dataframe and top1_target_recommendation
            task_recommendation = _self.task_selection(df, top1_target_recommendation)

            # generate appropriate explanation for the target column
            explanation_target_column = "Top recommendations (by Azure OpenAI API) for target colums are: ["+', '.join(column_inverse_map_keys)+"]"
            st.session_state.expltg = explanation_target_column

            # return recommended columns, top1 target column recommendation, and task recommendation
            return recommended_column_names, top1_target_recommendation, task_recommendation
        except:

            # in case of exception, display error and ask user to manually select target column 
            err = json.loads(curl_execution.stdout)
            st.write("LLM API call failed due to following error, please manually select the target column and task type")
            st.write(err)

            # reinitialize some parameters for the next run
            st.session_state.expltg = ""
            recommended_column_names = input_column_names
            top1_target_recommendation = None
            task_recommendation = None

            # return recommended columns, top1 target column recommendation, and task recommendation
            return recommended_column_names, top1_target_recommendation, task_recommendation


    # if user makes any changes in target column, task type should change accordingly
    def predict_tasktype_onchange_targetcol(_self, df):
        st.session_state.tt = ""
        st.session_state.expltt = None

        if st.session_state.tc:
            st.session_state.tt=_self.task_selection(df, st.session_state.tc)

    # main function which will call column_selection and task_selection
    def column_selection_task_selection(_self, csv_dataset): 
        
        with st.spinner('Identifying Target Column and ML Task Type...'):

            colnames = list(csv_dataset.columns)
            tasklist = ["","Classification", "Regression"]

            # if there are more than 10000 columns in the dataset, user has to manually select target column
            if len(colnames)>10000:
                st.warning("The dataset has over 10000 columns columns, please select the target column manually.")
                recommended_target_columns = st.multiselect("Target Columns", colnames, key="tc", on_change=_self.predict_tasktype_onchange_targetcol, args=(csv_dataset,))
                recommended_ml_task = st.selectbox("What is the ML task?", tasklist, index=0, key="tt")
            else:
                st.warning("Recommended target column and ML task type for the uploaded dataset will be selected below. Please verify the correctness and add more target columns if needed.")
                input_column_names = []
                colmap = {}
                for c in colnames:
                    if not c.lower().startswith('unnamed'):
                        res = c.strip()
                        colmap[res] = c
                        input_column_names.append(res)
                    
                # initialize session state variables to hold explanations for the target column and task type
                if 'expltg' not in st.session_state:
                    st.session_state.expltg = None
                if 'expltt' not in st.session_state:
                    st.session_state.expltt = None

                # call column_selection function and populate multiselect (dropdown) in the UI 
                recommended_column_names, top1_target_recommendation, stind = _self.column_selection(csv_dataset, input_column_names, colmap)
                recommended_target_columns = st.multiselect("Target Columns", recommended_column_names, top1_target_recommendation, key="tc", on_change=_self.predict_tasktype_onchange_targetcol, args=(csv_dataset,))
                st.write(recommended_target_columns)
                recommended_ml_task = st.selectbox("What is the ML task?", tasklist, index=tasklist.index(stind), key="tt")

                # display explanations for the target column and task type in the UI
                if st.session_state.expltg is not None and st.session_state.expltt is not None:
                    st.info("Explanations", icon="🗒️")
                    st.write(st.session_state.expltg)
                    st.write(st.session_state.expltt)
                
        return recommended_target_columns, recommended_ml_task
