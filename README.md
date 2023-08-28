# Synthetic Times Series Generation

## Context

Modelling in finance is a challenging task: the data often has complex statistical properties and its inner workings are largely unknown. Deep learning algorithms are making progress in the field of data-driven modelling, but the lack of sufficient data to train these models is currently holding back several new applications. Generative Adversarial Networks (GANs) are a neural network architecture family that has achieved good results in image generation and is being successfully applied to generate time series and other types of financial data.  
The purpose of this project is to use GANs to generate synthetic time series of S&P500 returns. Then we will train a LSTM model on the synthetic returns to see whether it performs better than a LSTM model trained on real returns.

# Results

## Time Series Generated

![image info](./images/time_series_generated.png)

We have generated a realistic time series !

## Time Series Prediction With LSTMs

### LSTM Trained On Synthetic Returns

![image info](./images/prediction_synthetic_data.png)

The root mean squared error is 31.5.

### LSTM Trained On Real Returns

![image info](./images/prediction_real_data.png)

The root mean squared error is 92.2.

## Conclusion

We achieved a 3x lower RMSE by training a LSTM model on our synthetic data than on real data!
