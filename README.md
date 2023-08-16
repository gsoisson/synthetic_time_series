# Synthetic Time Series

We will use GANs to generate synthetic financial time series. We will train a GAN on a dataset of S&P500 returns, and then train an LSTM model on the synthetic time series we generated, to see whether it performs as well as an LSTM model trained on a dataset of real S&P500 returns.
