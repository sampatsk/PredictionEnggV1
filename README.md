# PredictionEnggV1

Prediction Engineering (Automatic Target Column Identification and ML Task Type Prediction)\
This repository contains only prediction engineering batchrun script for benchmarking its performance with respect to a large testing dataset and not the complete AutoDW code. 

For AutoDW demo, visit http://autodw.fla.fujitsu.com/Fujitsu_Auto_Data_Wrangling page\
For AutoDW code, visit https://dlvault.fla.fujitsu.com/lliu/Data_wrangling_LLM/tree/main/data_wrangling_llm_code page

# Test dataset for benchmarking

- Clone this repository
- Download test dataset from this link (275 csv files) - https://fujitsunam-my.sharepoint.com/:u:/g/personal/ssampat_fujitsu_com/EW-oPqoubRBOhOFzHAwYJw0BrfNNyhd_us33fUW3KDZaiA?e=YZoqXc
- Extract the downloaded dataset inside this repository
- ```GT_target_task.json``` file (provided in this repository) contains ground-truth target column and ML task type for each dataset, against which accuracy will be calculated.

# Exception Cases

Testing on the following datasets will be ignored due to certain assumptions with current AutoDW platform:

(I) Files with more than 10000 columns (due to LLM token limit and streamlit demo being too slow, user has to do manual selection)
- 1128.csv

(II) Files larger than 1GB in size (streamlit demo limits files size to 1GB currently)
- austinreese_craigslist-carstrucks-data_vehicles.csv
- Corporacion-Favorita-Grocery-Sales-Forecasting.csv
- g-research-crypto-forecasting_train.csv
- Microsoft-Malware-Prediction.csv
- predict-closed-questions-on-stack-overflow_train.csv

(III) Files with more than one ground-truth target columns (LLM based target prediction only returns top-1 result)
- enb.csv
- jura.csv
- kevinmh_fifa-18-more-complete-player-dataset_complete.csv
- see-click-predict-fix_train.csv
- sf1.csv
- sf2.csv

# Run batch script

Follow entire installation instructions mentioned on the "Getting Started" section of https://dlvault.fla.fujitsu.com/lliu/Data_wrangling_LLM/tree/main/data_wrangling_llm_code page\
(This will help you to create conda environment, install necessary libraries and set up LLM endpoint/key as environment variable)\
```cd PredictionEnggV1```\
```conda activate <name_of_env>```\
```python PE_<date>_<year>.py```

Running this script will iterate over all datasets and store prediction results in the respective ```results_<date>_<year>.csv``` file.

Column of result files are as follws:
- dataset name - string
- predicted target column (top1) - string
- GT target column - string
- correct target column (top1) - Boolean\
[True if GT target col == predicted target col (top1)]
- predicted target column (top3) - List of strings
- correct target column (top3) - Boolean\
[True if GT target col in predicted target col (top3)]
- predicted task type - ["Classification", "Regression"]
- GT task type - ["Classification", "Regression"]
- correct task type - Boolean\
[True if GT task type == predicted task type]
- multiple GT target column - Boolean\
[True if len(GT target column) > 1; if True, the dataset will be ignored from evaluation as we do not currently support this]

# Evaluation Metrics
- TC (top1_acc) = #datasets where correct target column (top1) is True / total #datasets
- TC (top3_acc) = #datasets where correct target column (top3) is True / total #datasets
- MLT \| TC (top1_acc) = #datasets where correct task type is True / total #datasets

# Results

| Version                            | LLM            | TC (top1_acc) | TC (top3_acc) | MLT \| TC (top1_acc) |
|------------------------------------|----------------|---------------|---------------|----------------------|
| PE_march19_2024 (v2 kozuchi)       | Azure OpenAI   | 0.84          | 0.91          | 0.85                 |
|                                    | Fujitsu ChatAI | 0.74          | 0.80          | 0.74                 |
| PE_may12_2024 (LLM Auth)           | Azure OpenAI   |               |               |                      |
|                                    | Fujitsu ChatAI |               |               |                      |
| PE_mayXX_2024 (user/system prompt) | Azure OpenAI   |               |               |                      |
|                                    | Fujitsu ChatAI |               |               |                      |

Note: This process can take some time as there are some delays (time.sleep) purposely added to prevent exceeding token limit of LLMs per minute.\
Note: This script relies on LLMs, therefore it is possible that there are deviations in the predictions (and hence performance reported above) across multiple runs despite temperature set to 0. 
