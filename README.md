# CNN Sample Code For 學長

## 完整步驟
1. 安裝Anaconda(可直接裝最新版)
2. 設定虛擬環境(避免套件的版本有問題時可重新安裝)
3. 下載Pytorch 範例
4. 安裝需要的套件(有些版本不對會導致程式無法運行)
5. 執行範例程式

## 安裝Anaconda
* 根據下列網站的步驟安裝Anaconda (不用與網站中的版本相同，直接安裝最新版，Python 也一樣，後面在虛擬環境中的版本才需要注意)
https://medium.com/python4u/anaconda%E4%BB%8B%E7%B4%B9%E5%8F%8A%E5%AE%89%E8%A3%9D%E6%95%99%E5%AD%B8-f7dae6454ab6

## 設定虛擬環境
* refor from : https://medium.com/python4u/%E7%94%A8conda%E5%BB%BA%E7%AB%8B%E5%8F%8A%E7%AE%A1%E7%90%86python%E8%99%9B%E6%93%AC%E7%92%B0%E5%A2%83-b61fd2a76566
* 先以管理員身分開啟命令提示字元(CMD)(後面步驟多在CMD下操作)
![image](https://github.com/wayne-byte/CNN-Sample-Code/blob/master/figure/CMD.png)

* 以下列指令透過conda 建立虛擬環境(Modify the name in myenv)(按y確認安裝)
```terminal
conda create --name myenv python=3.7
conda activate myenv
```
![image](https://github.com/wayne-byte/CNN-Sample-Code/blob/master/figure/env.png)

## 下載Pytorch 範例
* refor from : https://github.com/yunjey/pytorch-tutorial
* 以下列指令進入資料要放的位置(Modify the path in mypath)
```terminal
cd mypath
```

* 以下列指令透過git 下載範例程式(我是下載源範例，所以圖與指令不同)
```terminal
git clone https://github.com/wayne-byte/CNN-Sample-Code.git
```

* 我們要使用的是其中"pytorch-tutorial/tutorials/04-utils/tensorboard/" 的部分
但不要照著他的方法安裝套件(因為他的安裝方法沒打版本，會導致各套件的衝突)，要以下列步驟安裝

## 安裝需要的套件
* 範例程式使用的套件是pytorch，但為了讓資料在顯示上可以更方便，所以有安裝tensorflow，只是為了使用其tensorboard的部分讓訓練過程及結果更好處理
```terminal
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
conda install tensorflow==1.14.0=mkl*
conda install opencv
```

* 安裝Spyder (python中最像Matlab的IDE)
```terminal
conda install spyder
```

## 執行範例程式
* 以下指令開啟spyder，並進入.\CNN-Sample-Code\pytorch-tutorial\tutorials\04-utils
```terminal
spyder
```

* 此範例是利用經典手寫辨識資料庫訓練模型做預測，細節可參考源網站所介紹，但由於版本關西對於檔案有些修改，可對照源版本的loger.py、 NN.py(main.py)、 CNN.py(main.py) 對照比較
* 開啟NN.py執行 (F5)
![image](https://github.com/wayne-byte/CNN-Sample-Code/blob/master/figure/NN.png)

* 開啟CNN.py執行 (F5)
![image](https://github.com/wayne-byte/CNN-Sample-Code/blob/master/figure/CNN.png)

* 從結果可看出在相同的輸入及訓練數下，CNN模型比NN模型的準確率要更高(嚴謹一點的比較是將2種模型的超參數(epoch、batch、layer等)都以很多組去訓練，取最好的模型才能比較性能)
