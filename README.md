# Report：Ground-Satellite-Geo-localization
## Introduction
We are excited to present the test dataset, designed for multi - weather cross - view geo - localization. This dataset aims to simulate real - world geo - localization scenarios, presenting new challenges for the research community.

This year's challenge in ground - satellite geo - localization has a specific focus. It aims to match partial street images to their corresponding satellite images. By concentrating on partial views, it intends to more accurately simulate real - world scenarios. In real life, obstructions or limited sensor angles can restrict the field of view, which is often encountered during low - altitude UAV operations for navigation, search - and - rescue missions, and autonomous flight.
The challenge utilizes the University - 1652 as the challenge dataset. This dataset offers 2,579 street images as queries and 951 gallery satellite images.

## Test 1
* **Training:** Train the model on the University - 1652 training set with 3 views (Drone+Satellite+Street).
* **Evaluation:** Evaluate University - 1652 test set to find the most similar satellite-view image to localize the target building in the satellite view.

## Test 2
* **Training:** Train the model on the University - 1652 training set with 3 views (Drone+Satellite+Street).  
* **Download:** Download the name - masked test set from the competition page (Onedrive link).  
* **Feature Extraction:** Extract features of the downloaded name - masked test set using the model trained in the first step.  
* **Evaluation:** Modify the demo.py or eveluate_gpu.py to compare features and save the top 10 image names in the gallery one by one. Ensure the test order is the same as the given query name text.  

## Datasets
* Download [University-1652] upon request.
* Download [name-masked test-160k-WX dataset (query & gallery+distractor)] from OneDrive

## Useful Links
* Basic Tutorial: [[https://github.com/layumi/University1652 - Baseline/tree/master/tutorial](https://github.com/layumi/University1652 - Baseline/tree/master/tutorial)](https://github.com/layumi/University1652-Baseline.git)
* Challenge Submission Site: https://codalab.lisn.upsaclay.fr/competitions/18770
* This repository contains the dataset link and the code for the paper University-1652: A Multi-view Multi-source Benchmark for Drone-based Geo-localization, ACM Multimedia 2020. The offical paper link is at https://dl.acm.org/doi/10.1145/3394171.3413896.
* Onedrive: https://hdueducn-my.sharepoint.com/personal/wongtyu_hdu_edu_cn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fwongtyu%5Fhdu%5Fedu%5Fcn%2FDocuments%2FDatasets%2Funiversity%5F160k%5Fwx%5Ftest%5Fset&ga=1

## Code Repository Structure
The code repository contains the following key components:
model.py: Defines neural network models such as two_view_net and three_view_net.
utils.py: Provides utility functions for making balanced class weights, loading and saving models.
image_folder.py: Defines custom dataset classes for handling different types of data, like CustomData160k_sat and CustomData160k_drone.
test_160k.py: Used for testing the model on the test dataset, including feature extraction and result ranking.
demo.py: Demonstrates how to visualize the ranking results for Test 1.
demo_160k.py: Demonstrates how to visualize the answer.txt results (Test 2).
