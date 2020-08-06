# Recognition-of-Chinese-Medical-Speech
本網站的程式經由參考並修改 https://https://github.com/Alexander-H-Liu/End-to-end-ASR-Pytorch 的內容獲得。



# 預處理

此網址提供一預處理程式，針對 Chinese Medical Speech Corpus (ChiMeS) 資料庫進行預處理，經處理後將便於後續的訓練與測試。

#### 資料集

需要先產生用於訓練/測試的檔案，格式為.csv 檔，檔案內每行的內容為一個音檔的路徑與對應的文字檔路徑，如下所示：
```
/dataset/audio/audio_1.wav,/dataset/label/label_1.txt
/dataset/audio/audio_2.wav,/dataset/label/label_2.txt
...
```

預處理有三個程式檔案可供使用，記得改裡面的檔案路徑。

1. generate_component.py

   產生預處理文字所需要的檔案

2. preprocess_data_train_test.py:

   將訓練集/測試集的語音轉成.npy及文字進行編碼

3. preprocess_data_one.py

   將額外想測試檔案的語音轉成.npy及文字進行編碼
