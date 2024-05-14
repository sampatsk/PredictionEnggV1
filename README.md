# PredictionEnggV1
Prediction Engineering (Automatic Target Column Identification and ML Task Type Prediction)\
This repository contains only prediction engineering part (scripts with streamlit UI and batchrun script for benchmarking) and not the complete AutoDW code. 

For AutoDW demo, visit http://autodw.fla.fujitsu.com/Fujitsu_Auto_Data_Wrangling page\
For AutoDW code, visit https://dlvault.fla.fujitsu.com/lliu/Data_wrangling_LLM/tree/main/data_wrangling_llm_code page

# Test dataset for benchmarking
- Clone this repository
- Download test dataset from this link (275 csv files) - https://fujitsunam-my.sharepoint.com/:u:/g/personal/ssampat_fujitsu_com/EW-oPqoubRBOhOFzHAwYJw0BrfNNyhd_us33fUW3KDZaiA?e=YZoqXc
- Extract the downloaded dataset inside this repository

# Test on following files will be ignored due to certain assumptions

Files with more than 10000 columns (due to LLM token limit and streamlit demo being too slow, so user has to do manual selection)
- 1128.csv

Files larger than 1G in size (streamlit demo limits files size to 1G currently)
- austinreese_craigslist-carstrucks-data_vehicles.csv
- Corporacion-Favorita-Grocery-Sales-Forecasting.csv
- g-research-crypto-forecasting_train.csv
- Microsoft-Malware-Prediction.csv
- predict-closed-questions-on-stack-overflow_train.csv

Files with more than one target columns (LLM based target prediction only predicts top-1 result by default)
- enb.csv
- jura.csv
- kevinmh_fifa-18-more-complete-player-dataset_complete.csv
- see-click-predict-fix_train.csv
- sf1.csv
- sf2.csv

GT_target_task.json file contains ground-truth target column and ML task type for each dataset, against which accuracy will be calculated\
Accuracy is calculated over 275 (total csvs) - 12 (above exceptions) = 263 count

# Run batch script

Follow entire installation instructions mentioned on the "Getting Started" section of https://dlvault.fla.fujitsu.com/lliu/Data_wrangling_LLM/tree/main/data_wrangling_llm_code page\
(to create conda environment and set up LLM endpoint/key as environment variable)\
```cd PredictionEnggV1```\
```conda activate <name_of_env>```\
```python PE_<date>_<year>.py```\

This will iterate over all datasets and store prediction results in respective ```results_<date>_<year>.csv``` file.

Column headers of result files are as follws:
- dataset name
- predicted target col (top1)
- GT target col (top1)
- correct target column (top1)
- predicted target col (top3)
- correct target column (top3)
- predicted task type
- GT task type
- correct task type
- multiple GT target cols

This process can take some time as there are some delays (time.sleep) purposely added to prevent exceeding token limit of LLMs per minute.

# Evaluation Metrics
TC (top1_acc) - # datasets where LLM predicted Top1 target column exactly matches with the ground-truth target column / total # datasers (i.e. 263)
TC (top3_acc) - # datasets where LLM predicted Top3 target column contains the ground-truth target column / total # datasers (i.e. 263)
MLT \| TC (top1_acc) - # datasets where predicted ML task type target column contains the ground-truth target column / total # datasers (i.e. 263)

# Results

| Version                            | LLM            | TC (top1_acc) | TC (top3_acc) | MLT \| TC (top1_acc) |
|------------------------------------|----------------|---------------|---------------|----------------------|
| PE_march19_2024 (v1 kozuchi)       | Azure OpenAI   | 0.84          | 0.91          | 0.85                 |
|                                    | Fujitsu ChatAI | 0.74          | 0.80          | 0.74                 |
| PE_may12_2024 (LLM Auth)           | Azure OpenAI   |               |               |                      |
|                                    | Fujitsu ChatAI |               |               |                      |
| PE_mayXX_2024 (user/system prompt) | Azure OpenAI   |               |               |                      |
|                                    | Fujitsu ChatAI |               |               |                      |
