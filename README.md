# Missing Keypoint imputation using Random forest regression

## Motivation:
Mobility assessment on elderly using keypoints on video: 
The motivation is to acquire the knee bend estimation and hip bend estimation values same as the expensive wearable sensors but using video analytics using keypoint estimators. It is to prove that video analytics could work same as wearable sensors without having the need for the patient to wear the sensors. 

## Big picture:
* Used 17 point human keypoint detector ([EfficientHRNet](https://arxiv.org/abs/2007.08090]) to get the keypoints.
* Calculate the knee bend angles and hip bend angle. Compare it to the readings from the wearable sensors .
* Used Random forest regressor to regress the missing keypoints offline.

## Missing Data imputation using Random Forest regressor



