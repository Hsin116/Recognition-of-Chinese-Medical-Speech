# Recognition-of-Chinese-Medical-Speech
本網站的程式經由參考並修改 https://https://github.com/Alexander-H-Liu/End-to-end-ASR-Pytorch 的內容獲得。



## 預處理

此網址提供針對 Chinese Medical Speech Corpus ([ChiMeS](https://ee303.nctu.me/home)) 資料庫進行預處理的程式，經處理後將便於後續的訓練與測試。

#### 資料集

需要先產生用於訓練/測試的檔案，格式為.csv 檔，檔案內每行的內容為一個音檔的路徑與對應的文字檔路徑，如下所示：
```
/dataset/audio/audio_1.wav,/dataset/label/label_1.txt
/dataset/audio/audio_2.wav,/dataset/label/label_2.txt
...
```

#### 預處理有三個程式檔案可供使用 (改裡面的檔案路徑):

1. generate_component.py

   產生預處理文字所需要的檔案

2. preprocess_data_train_test.py:

   將訓練集/測試集的語音轉成.npy及文字進行編碼

3. preprocess_data_one.py

   將額外想測試檔案的語音轉成.npy及文字進行編碼


## 建立 Config 檔案

與訓練/測試有關的所有參數都將存儲在yaml文件中。可參考example.yaml以獲取參數格式，此參數需要自行修改以獲得最佳效能。


## 訓練

當準備好 config 及資料預處理完後，就可以開始進行訓練。
訓練的命令如下:
```
python3 main.py --config /Joint/new_exp/example.yaml --logdir /Joint/new_exp/log --ckpdir /Joint/new_exp/model --gpu-num 0 --spec-aug
```
#### 上述參數中:
* config : config 的 .yaml 檔之路徑
* logdir : log儲存之路徑
* ckpdir : 模型權重儲存之路徑
* gpu-num : 若有多顆 gpu 則可以指定 gpu
* spec-aug : 使用頻譜增量


## 測試

測試階段將訓練時所儲存的各個 epoch 的模型權重檔對測試集的語音進行預測，並與ground truth 計算 Character error rate (CER) 作為評判模型好壞的標準。
訓練的命令如下:
```
python3 main.py --config /Joint/new_exp/example.yaml --ckpdir /Joint/new_exp/model --C2E /Joint/new_exp/C2E.json --test --gpu-num 0
```
#### 上述參數中:
* config : config 的 .yaml 檔之路徑
* ckpdir : 模型權重儲存之路徑
* C2E : 用於將英文音節轉換為中文字的 .josn 檔
* gpu-num : 若有多顆 gpu 則可以指定 gpu
* test : 開啟測試階段
