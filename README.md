# A New Modified Deep Convolutional Neural Network for detecting COVID-19 from X-ray images

COVID-19 has become a serious health problem all around the world.  It is confirmed that this virus has taken over 126,607 lives until today. Since the beginning of its spreading, many Artificial Intelligence researchers developed systems and methods for predicting the virus's behavior or detecting the infection. One of the possible ways of determining the patient infection to COVID-19 is through analyzing the chest X-ray images. As there are a large number of patients in hospitals, it would be time-consuming and difficult to examine lots of X-ray images, so it can be very useful to develop an AI network that does this job automatically.  In this paper, we have trained several deep convolutional networks with the introduced training techniques for classifying X-ray images into three classes: normal, pneumonia, and COVID-19, based on two open-source datasets. Unfortunately, most of the previous works on this subject have not shared their dataset, and we had to deal with few data on covid19 cases. Our data contains 180 X-ray images that belong to persons infected to COVID-19, so we tried to apply methods to achieve the best possible results. In this research, we introduce some training techniques that help the network learn better when we have few cases of COVID-19, and also we propose a neural network that is a concatenation of Xception and ResNet50V2 networks. This network achieved the best accuracy by utilizing multiple features extracted by two robust networks. In this paper, despite some other researches, we have tested our network on 11302 images to report the actual accuracy our network can achieve in real circumstances. The average accuracy of the proposed network for detecting COVID-19 cases is 99.56%, and the overall average accuracy for all classes is 91.4%.

The two open-source datasets are available on:

1- https://github.com/ieee8023/covid-chestxray-dataset

2-https://www.kaggle.com/c/rsna-pneumonia-detection-challenge 

The first dataset contains COVID-19 and some other diseases like ARDS, SARS, Streptococcus, Pneumocystis.

The second dataset contains patients with pneumonia and normal people.

Some of the images of these datasets are:

<img src="/images/normal.png" width="30%"> <img src="/images/pneumonia.png" width="30%"> <img src="/images/covid.png" width="30%">

 *These images show Normal, Pneumonia and COVID-19 cases from left to right respectively.*


We have used a concatenation of ResNet50V2 and Xception networks as the network for classifying the images into 
three classes: Normal, Pneumonia, and COVID-19.

<p align="center">
	<img src="images/concatenated_net.png" alt="photo not available" width="100%" height="70%">
	<br>
	<em>The architecture of our proposed network</em>
</p>
 Dataset | COVID-19 | Pneumonia | Normal
------------ | ------------- | ------------- | -------------
covid chestxray dataset | 180 | 42 | 0
rsna pneumonia detection challenge | 0 | 6012 | 8851
Total | 180 | 6054 | 8851
All Training Sets | 149 | 1634 | 2000
Validation Set | 31 | 4420 | 6851

As it is stated, we only had 180 cases infected to COVID-19, which is almost a few numbers of data for a class. If we
had combined lots of images from normal or pneumonia classes with few images of COVID-19 class, the network
would become able to detect pneumonia and normal classes very well, but not the COVID-19 class. In that case, as
there are many more images of pneumonia and normal classes than the COVID-19 class, the general accuracy would
become very high, not the COVID-19 detection accuracy, which is not our goal because the main purpose here is to
achieve good results in detecting COVID-19 cases and not detecting false COVID-19 cases.
We decided to select 250 random cases of normal class and 234 random cases of pneumonia class along with the 149
COVID-19 cases for training. In total, we had 633 cases for the training set. We used another method for also improving
the general detection accuracy. In this method, we selected eight different training sets with 633 cases that 149 of
them were COVID-19 cases and 34 pneumonia cases from the first dataset that is common between all sets. Each
set contained 250 normal and 200 pneumonia cases that are unique. Based on this categorizing, our training set than
includes eight sets is made of 3783 images. By doing so, we show different normal and pneumonia cases to the network
in each training phase along with the same COVID-19 cases. This makes the network better be able to distinguish the
COVID-19 cases from the other ones, so the rate of false detected COVID-19 cases decreases. This condition also
causes the network to learn other classes features better, so the general accuracy will increase.
We allocated 8 phases for training, which in each training phase one train set was fed to the network for training during
100 epochs; in total, we trained the network for 800 epochs. For reporting more accurate results, we chose five folds for
training, which in every fold the training set was made of 8 phases as it is mentioned.

We have evaluated our network on 11302 images to show the real performance of our proposed network.

The confusion matrixes for two folds are depicted below:

<img src="/Confusion_matrix/concatenate-fold1-confusion_matrix-1.jpg" width="50%"><img src="/Confusion_matrix/concatenate-fold3-confusion_matrix-1.jpg" width="50%">


 The next tables will show the average of specificity and recall metrics for each class between five folds.
 
 Metric | COVID-19 | Pneumonia | Normal
------------ | ------------- | ------------- | -------------
Specificity | 99.56 | 94.32 | 88.09
Accuracy | 99.50 | 91.60 | 91.71

You can access my written codes here as follows:

In the data Loading-Training-Evaluating.ipynb file, you can find our codes for loading data, network training, and evaluation.

The dataset preparing.ipynb file contains codes that were used for preparing the dataset, and some part of this code is inspired by Linda Wang and Alexander Wong work that is shared on https://github.com/lindawangg/COVID-Net/blob/master/README.md.

Results.py includes the codes that are written to outputs the confusion matrixes and details for the tables.

You can also access and use all of our trained networks for each fold in :
https://drive.google.com/drive/folders/19R4T-D-bWUkQOh3xy5CkIDAmkLBt8ID7?usp=sharing

In the results folder, you can access all the details of our achieved results. The Confusion_matrix folder is included the confusion matrixes of all the networks we tested for each fold, and in the prepared_csv_files directory, we have shared all the CSV files we generated and used in our work.
