# PredictionEnggV1
Prediction Engineering (Automatic Target Column Identification and ML Task Type Prediction)\
This repository contains only prediction engineering part (scripts with streamlit UI and batchrun script for benchmarking) and not the complete AutoDW code. 

For AutoDW demo, visit http://autodw.fla.fujitsu.com/Fujitsu_Auto_Data_Wrangling

For AutoDW code, visit https://dlvault.fla.fujitsu.com/lliu/Data_wrangling_LLM/tree/main/data_wrangling_llm_code

# Test dataset for benchmarking
Download test dataset from this link (275 csv files) - https://fujitsunam-my.sharepoint.com/:u:/g/personal/ssampat_fujitsu_com/EW-oPqoubRBOhOFzHAwYJw0BrfNNyhd_us33fUW3KDZaiA?e=YZoqXc

# Test on following files will be ignored due to certain assumptions

Files with more than 5000 columns (streamlit demo does not support)
-
-
-
-
-

Files larger than 1G in size (streamlit demo does not support)
- austinreese_craigslist-carstrucks-data_vehicles.csv
- Corporacion-Favorita-Grocery-Sales-Forecasting.csv
- g-research-crypto-forecasting_train.csv
- Microsoft-Malware-Prediction.csv
- predict-closed-questions-on-stack-overflow_train.csv

# Results

| Version                             | LLM            | TC (top1_acc) | TC (top3_acc) | MLT |
|-------------------------------------|----------------|---------------|---------------|-----|
| PE_march19_2024 (v1 kozuchi)        | Azure OpenAI   |               |               |     |
|                                     | Fujitsu ChatAI |               |               |     |
| PE_may12_2024  (LLM Auth)           | Azure OpenAI   |               |               |     |
|                                     | Fujitsu ChatAI |               |               |     |
| PE_mayXX_2024  (user/system prompt) | Azure OpenAI   |               |               |     |
|                                     | Fujitsu ChatAI |               |               |     |
