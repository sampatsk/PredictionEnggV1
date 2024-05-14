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

Files with more than 5000 columns (due to LLM token limit and streamlit demo too slow, so user has to do manual selection)
- 
-
-
-
-

Files larger than 1G in size (streamlit demo limits files size to 1G currently)
- austinreese_craigslist-carstrucks-data_vehicles.csv
- Corporacion-Favorita-Grocery-Sales-Forecasting.csv
- g-research-crypto-forecasting_train.csv
- Microsoft-Malware-Prediction.csv
- predict-closed-questions-on-stack-overflow_train.csv

# Run batch script

Follow entire installation instructions mentioned on the "Getting Started" section of https://dlvault.fla.fujitsu.com/lliu/Data_wrangling_LLM/tree/main/data_wrangling_llm_code page\
(to create conda environment and set up LLM endpoint/key as environment variable)\
```cd PredictionEnggV1```\
```conda activate <name_of_env>```\
```python PE_<date>_<year>.py```\
This will iterate over all datasets and store prediction results in respective results_<date>_<year>.csv file.

This process can take some time as there are some delays (time.sleep) purposely added to prevent exceeding token limit of LLMs per minute.

# Results

| Version                             | LLM            | TC (top1_acc) | TC (top3_acc) | MLT |
|-------------------------------------|----------------|---------------|---------------|-----|
| PE_march19_2024 (v1 kozuchi)        | Azure OpenAI   |               |               |     |
|                                     | Fujitsu ChatAI |               |               |     |
| PE_may12_2024 (LLM Auth)            | Azure OpenAI   |               |               |     |
|                                     | Fujitsu ChatAI |               |               |     |
| PE_mayXX_2024 (user/system prompt)  | Azure OpenAI   |               |               |     |
|                                     | Fujitsu ChatAI |               |               |     |
