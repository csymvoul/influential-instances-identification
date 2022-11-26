# Import libraries 
import numpy as np

"""
    The steps needed to train an AI model based on the influential instances are:
        1. Train the model on the original dataset
        2. Calculate the DFBETA and RMSE values for each instance: 
            DFBETA_i = beta - beta_{-i}, where beta is the weight of the feature and beta_{-i} is the weight of the feature without the i-th instance
                The beta_{-i} is calculated by training the model on the dataset without the i-th instance
            RMSE_i = sqrt(1/n * sum(y_i - y_hat_i)^2), where y_i is the true value of the i-th instance and y_hat_i is the predicted value of the i-th instance
        3. Once the DFBETA and RMSE values are calculated, the influential instances are the ones with the highest DFBETA and lowest RMSE values
        4. Then only the influential instances are kept from the original dataset
        5. Consequently, we apply the K-Means algorithm, where K is equal to the number influential instances and we apply it to the excluded data of the original dataset
        6. The centroids of the K-Means algorithm are the influential instances
        7. We cluster the original dataset based on the centroids of the K-Means algorithm and we produce a new final dataset, which we use to re-train the model 
"""

# Step 1: Train the model on the original dataset


# Step 2: Calculate the DFBETA and RMSE values for each instance
# This steps requires re-training the original model on the dataset without the one instance at a time, in order to calculate its significance

