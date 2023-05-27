# README

File：余雪華_2023_dcard_ml_intern_homework.zip



## Prepare Environment
建議執行環境: Python 3.9.12、conda 22.9.0

建議執行在新環境且使用conda package，並安裝所需要的套件 requirement.txt

建議示例如下：
1. 為你的系統安裝Conda package，安裝的套件和方法可以參考[此處](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
2. 建立一個專屬該模型XHY_Dcard_intern_hw的Conda環境
```
conda create -n "XHY_Dcard_intern_hw" python=3.9.12
```
3. Activate XHY_Dcard_intern_hw Conda environment. 
```
conda activate XHY_Dcard_intern_hw
```

4. 如果希望離開此Conda environment，請輸入以下指令
```
conda deactivate XHY_Dcard_intern_hw
```
## Steps to Use
1. 解壓縮檔案：余雪華_2023_dcard_ml_intern_homework.zip，內含所需的套件、文件，。

2. 改變工作路徑
```
cd 余雪華_2023_dcard_ml_intern_homework/余雪華_Dcard_intern_homework/
```
3. 安裝所需套件
```
pip install -r requirements.txt
```
4. 確認訓練及測試檔案儲存位置，位於dataset folder
```
./dataset/ - intern_homework_train_dataset.csv
           - intern_homework_public_test_dataset.csv
           - intern_homework_private_test_dataset.csv
    
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

