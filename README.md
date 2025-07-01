# Sportsball Classification - README

## Overview

This project involves training and evaluating classification models for identifying different types of sports balls from image data under varying conditions of color representation. The tasks are split into three main parts, each building on the previous to explore model generalization and robustness.

---

## Part 1: Classification with Original Colors

- **Data Generation**:  
  Train and test data are generated with the original color settings (`--task4` commented out in `config.ini`).  
  - Image size: 32x32 pixels  
  - Object size: Max 24 pixels  
  - Classes: 4 types of sports balls  
  - Training samples: ~10,000  
  - Test samples: 1,000

- **Model Training**:  
  A simple convolutional neural network is trained on the generated data.

- **Evaluation**:  
  The trained model is tested on the 1,000 test images to evaluate its performance.

---

## Part 2: Random Colormap Transfer and Evaluation

- **Data Generation**:  
  Train and test data are generated using random colormaps (`--task4` enabled in `config.ini`).

- **Subtask A**:  
  - Use the model trained in Part 1 (no fine-tuning) on the mixed-color test set.  
  - **Objective**: Assess the generalization ability of the model to unseen color distributions.

- **Subtask B**:  
  - Finetune the model using 100 mixed-color images.  
  - Re-evaluate on the mixed-color test set.

- **Subtask C**:  
  - Train a new model from scratch using the full mixed-color training set.  
  - Evaluate on the mixed-color test set.

- **Subtask D**:  
  - Use the newly trained model from Subtask C to evaluate on the original-color test set (no fine-tuning).  
  - Finetune on 100 original-color images and test again.

- **Comparison and Discussion**:  
  Analyze the differences in generalization, performance drop due to color variation, and the effects of limited fine-tuning.

---

## Part 3: Multi-Label Classification

- **Code Modification**:  
  Update `data/sportballs.py` to include both:
  - Type of sportsball
  - Colormap type (original, inverted, grayscale)

- **Data Generation**:  
  Generate training and test data with random colormaps, annotated with dual labels.

- **Model Training**:  
  Train a model to predict both labels jointly (multi-task learning).

- **Evaluation and Discussion**:  
  Assess how well the model handles both tasks and discuss the interplay between the two outputs.

---

## Reporting

In addition to the above results and evaluations, include the following in your report as per the provided instructions in `ReadMe.txt`:

- Detailed discussion of results for each part
- Comparative analysis of model performances across different scenarios
- Insights into model generalization and limitations

