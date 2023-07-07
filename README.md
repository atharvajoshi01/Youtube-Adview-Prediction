# YouTube Adview Prediction
## Project Overview
This project focuses on predicting the number of adviews on YouTube videos based on various metrics and attributes such as views, likes, dislikes, comments, duration, and category. The goal is to build a regression model that accurately predicts the adview count, allowing advertisers to make informed decisions and optimize their marketing strategies.

## Dataset
The dataset used for this project is available in the train.csv file, which contains information about approximately 15,000 YouTube videos. It includes attributes such as vidid (unique identification ID for each video), adview (number of adviews), views, likes, dislikes, comments, published date, duration, and category of each video. The dataset is already provided in the GitHub repository along with the project files.

## Steps and Tasks
Import the necessary libraries and load the dataset:

1. Use popular libraries like numpy, pandas, matplotlib, seaborn, scikit-learn, and keras.
2. Load the dataset using pandas and explore its shape and data types.

## Visualize the dataset:

1. Use matplotlib and seaborn to create visually appealing plots.
2. Plot distributions, such as histograms and boxplots, to understand the data distributions for each attribute.
3. Create a heatmap to visualize the correlations between different features.
4. Identify and remove outliers using appropriate techniques.
   
### Data Cleaning:

1. Handle missing values by applying techniques like imputation or deletion.
2. Remove irrelevant or unimportant features that do not contribute significantly to the prediction task.

### Data Transformation:

1. Convert categorical attributes into numerical values using techniques like label encoding or one-hot encoding.
2. Perform necessary transformations, such as converting date and time attributes into suitable formats.
3. Utilize feature engineering techniques to create new features that may enhance the model's performance.

### Data Normalization and Splitting:

1. Normalize the numerical attributes to ensure all features are on a similar scale.
2. Split the dataset into training, validation, and test sets in an appropriate ratio, such as 80:10:10 or 70:15:15.

### Model Training and Evaluation:

1. Train various regression models, including linear regression, support vector regressor, decision tree regressor, random forest regressor, and artificial neural networks.
2. Use scikit-learn library to fit the models on the training data and calculate error metrics (e.g., mean squared error, mean absolute error) to evaluate their performance.

### Decision Tree Regressor and Random Forest Regressor:

1. Import decision tree regressor and random forest regressor models from the scikit-learn library.
2. Configure appropriate hyperparameters for these models.
3. Train the models using the training data and evaluate their performance by calculating error metrics.

### Artificial Neural Network:

1. Utilize the Keras library to build an artificial neural network.
2. Define the model architecture, including the number of layers, neurons, activation functions, and optimization algorithm.
3. Train the neural network on the training data and evaluate its performance using error metrics.
4. Experiment with different architectures and hyperparameters to improve the model's performance.

### Model Selection:

1. Compare the error metrics and generalization performance of different models.
2. Select the model with the lowest error and good generalization performance on the validation set.
3. Utilize evaluation metrics, such as F1 score and ROC curves, to assess the model's performance.

### Save the Model and Make Predictions:

1. Save the selected model for future use using appropriate functions or methods.
2. Make predictions on the test set using the saved model to estimate the number of adviews.

## Tools and Libraries Used

1. Python: Programming language used for data analysis and machine learning.
2. NumPy: Library for numerical operations.
3. pandas: Library for data analysis and manipulation.
4. Matplotlib: Library for data visualization.
5. seaborn: Library for statistical data visualization.
6. scikit-learn: Library for machine learning models and evaluation.
7. Keras: Deep learning library for building neural networks.
   
## Conclusion
By implementing various regression models and artificial neural networks on the YouTube adview dataset, we can accurately predict the number of adviews for YouTube videos. The selected model can assist advertisers in making data-driven decisions and optimizing their marketing strategies.

For a detailed implementation and analysis of the project, please refer to the notebook provided.
