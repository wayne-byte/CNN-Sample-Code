# CNN Sample Code

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
* refer from : https://medium.com/python4u/%E7%94%A8conda%E5%BB%BA%E7%AB%8B%E5%8F%8A%E7%AE%A1%E7%90%86python%E8%99%9B%E6%93%AC%E7%92%B0%E5%A2%83-b61fd2a76566
* 先以管理員身分開啟命令提示字元(CMD)(後面步驟多在CMD下操作)
![image](https://github.com/wayne-byte/CNN-Sample-Code/blob/master/figure/CMD.png)

* 以下列指令透過conda 建立虛擬環境(Modify the name in myenv)(按y確認安裝)
```terminal
conda create --name myenv python=3.7
conda activate myenv
```
![image](https://github.com/wayne-byte/CNN-Sample-Code/blob/master/figure/env.png)

## 下載Pytorch 範例
* refer from : https://github.com/yunjey/pytorch-tutorial
* 以下列指令進入資料要放的位置(Modify the path in mypath)
```terminal
cd mypath
```

* 以下列指令透過git 下載範例程式(我是下載源範例，所以上圖與指令不同)
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

## 討論
* 可參考源範例的READNE(https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/04-utils/tensorboard)，了解tensorboard 的使用
* 若要將範例套在學長的應用上，需要修改的主要是輸入的形式及模型的輸入大小
* 因模型的輸入大小需要固定，所以建議學長影像需要先做前處理(ex.先以整張影像輸入為目標，不需要的部分以補0處理、ex.影像若太大會導致batch需要設定很小，所以先做空間降採樣或PCA等等)

---

# 修改輸入 (8/31)
* refer from : https://www.cnblogs.com/denny402/p/7512516.html
* refer from : https://medium.com/@rowantseng/pytorch-%E8%87%AA%E5%AE%9A%E7%BE%A9%E8%B3%87%E6%96%99%E9%9B%86-custom-dataset-7f9958a8ff15

## 完整步驟
1. 將MNIST資料集改變大小後以影像檔輸出
2. 執行修改後範例程式

## 將MNIST資料集改變大小後以影像檔輸出
* 開啟Spyder後，進入.\CNN-Sample-Code\pytorch-tutorial\tutorials\MakeDataset，接下來的程式及資料皆在此路徑內
* 執行data_save.py，此程式將原本Pytorch的MNIST資料由原本28x28大小轉成64x64，並以圖片檔形式輸出到data_img/mnist_train及data_img/mnist_test內，並在data_img中產生mnist_train.csv紀錄訓練資料的路徑及對應的label
![image](https://github.com/wayne-byte/CNN-Sample-Code/blob/master/figure/data.png)

## 執行修改後範例程式
* 依下列指令安裝套件，此套件沒啥意義，只是讓進度條更方便
```terminal
conda install -c conda-forge tqdm
```

* 執行MNIST_CNN.py，由於圖片只是內插放大，並沒有增加資訊量，所以結果與之前不會有太大差異
![image](https://github.com/wayne-byte/CNN-Sample-Code/blob/master/figure/MakeData_CNN.png)

* 可與之前的MNIST_CNN.py做對比，改以Mydataset去讀取圖片檔，且由於輸入大小有變所以模型的結構也有些變化

## 討論
* 若要以學長的資料作為輸入，在資料部分需要將data_img/mnist_train及data_img/mnist_train中的檔案換成學長的資料，且改變mnist_train.csv及mnist_test.csv中第二行的label即可
* 但由於輸入大小有變，所以模型的結構也需要修改，而修改模型需要遵守維度的變化，可參考以下網站
* https://chtseng.wordpress.com/2017/09/12/%E5%88%9D%E6%8E%A2%E5%8D%B7%E7%A9%8D%E7%A5%9E%E7%B6%93%E7%B6%B2%E8%B7%AF/
* https://hackmd.io/@shaoeChen/BJDUj508z?type=view#1-11Why-convolutions

