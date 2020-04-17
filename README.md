# A New Modified Deep Convolutional Neural Network for detecting COVID-19 from X-ray images

COVID-19 has become a serious health problem all around the world.  It is confirmed that this virus has taken over 126,607 lives until today. Since the beginning of its spreading, many Artificial Intelligence researchers developed systems and methods for predicting the virus's behavior or detecting the infection. One of the possible ways of determining the patient infection to COVID-19 is through analyzing the chest X-ray images. As there are a large number of patients in hospitals, it would be time-consuming and difficult to examine lots of X-ray images, so it can be very useful to develop an AI network that does this job automatically.  In this paper, we have trained several deep convolutional networks with the introduced training techniques for classifying X-ray images into three classes: normal, pneumonia, and COVID-19, based on two open-source datasets. Unfortunately, most of the previous works on this subject have not shared their dataset, and we had to deal with few data on covid19 cases. Our data contains 180 X-ray images that belong to persons infected to COVID-19, so we tried to apply methods to achieve the best possible results. In this research, we introduce some training techniques that help the network learn better when we have few cases of COVID-19, and also we propose a neural network that is a concatenation of Xception and ResNet50V2 networks. This network achieved the best accuracy by utilizing multiple features extracted by two robust networks. In this paper, despite some other researches, we have tested our network on 11302 images to report the actual accuracy our network can achieve in real circumstances. The average accuracy of the proposed network for detecting COVID-19 cases is 99.56%, and the overall average accuracy for all classes is 91.4%.

The two open-source datasets are available on:

1- https://github.com/ieee8023/covid-chestxray-dataset

2-https://www.kaggle.com/c/rsna-pneumonia-detection-challenge 

The first dataset contains COVID-19 and some other diseases like: ARDS, SARS, Streptococcus, Pneumocystis.

The second dataset contains patients with pneumonia and normal people.

Some of the images of these datasets are:

{% include image.html img="/images/normal.png"  caption="dd"}
<img src="/images/normal.png" width="30%">
<img src="/images/pneumonia.png" width="30%"> <img src="/images/covid.png" width="30%">
 

