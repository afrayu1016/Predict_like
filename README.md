# README


## Prepare Environment
建議執行環境: Python 3.9.12、conda 22.9.0

建議執行在新環境且使用conda package，並安裝所需要的套件 requirement.txt

建議示例如下：
1. 為你的系統安裝Conda package，安裝的套件和方法可以參考[此處](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
2. 建立一個專屬該模型XHY_Dcard_intern_hw的Conda環境
```
conda create -n "your_env" python=3.9.12
```
3. Activate "your_env" Conda environment. 
```
conda activate "your_env"
```

4. 如果希望離開此Conda environment，請輸入以下指令
```
conda deactivate "your_env"
```
## Steps to Use
1. git clone https://github.com/afrayu1016/predict_llike.git

2. 安裝所需套件
```
pip install -r requirements.txt
```
3. 確認訓練及測試檔案儲存位置，位於dataset folder
```
./dataset/ - intern_homework_train_dataset.csv
           - intern_homework_public_test_dataset.csv
           - intern_homework_private_test_dataset.csv
    
```
4. 安裝ckiptagger 套件於 ./data下
參考 https://github.com/ckiplab/ckiptagger/wiki/Chinese-README
```
./data/ - embedding_character/
           - token_list.npy
           - vector_list.npy
        - embedding_word/
           - token_list.npy
           - vector_list.npy
        - LICENSE
        - model_ner/
           - label_list.txt
           - model_ontochinese_Att-0_BiLSTM-2-500_batch128-run1.data-00000-of-00001
           - model_ontochinese_Att-0_BiLSTM-2-500_batch128-run1.index
           - model_ontochinese_Att-0_BiLSTM-2-500_batch128-run1.meta
           - pos_list.txt
        - model_pos/
           - label_list.txt
           - model_asbc_Att-0_BiLSTM-2-500_batch256-run1.data-00000-of-00001
           - model_asbc_Att-0_BiLSTM-2-500_batch256-run1.index
           - model_asbc_Att-0_BiLSTM-2-500_batch256-run1.meta
        - model_ws/
           - model_asbc_Att-0_BiLSTM-cross-2-500_batch128-run1.data-00000-of-00001
           - model_asbc_Att-0_BiLSTM-cross-2-500_batch128-run1.index
           - model_asbc_Att-0_BiLSTM-cross-2-500_batch128-run1.meta
    
```

>File descriptions:
> * intern_homework_train_dataset.csv: 共50000筆訓練資料，內含真實like_count_24h數值
>  * intern_homework_public_test_dataset.csv: 共10000筆公開測試資料，內含真實like_count_24h數值
>  * intern_homework_private_test_dataset.csv: 共10000筆非公開測試資料，不含真實like_count_24h數值
6. Run XHY_Dcard_intern_hw 並預測private test data結果，此結果會儲存在_output folder。
```
python XHY_Dcard_intern_hw.py 
```


## Example and Output Results
**SAMPLE OUTPUTS**
```
./_output/ - result.csv
```
>File descriptions:
> * result.csv: 預測private test set的like_count_24h欄位的結果  

result.csv 檔案內容示例：
```
like_count_24h
78.999
```

