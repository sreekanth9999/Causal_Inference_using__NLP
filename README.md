# Causal_Inference_using__NLP

The task here is to analyse the sentiment of the stock price of the IPO companies by analysing the importance of text published in the company's 10K documents using double machine learning(de-biased machine learning). 

Initially I have extracted the 10K documents of thousands of IPO companies from WRDS. 
Then we extracted different sections from the text and measured different types of similarities between 
these texts to form our predictor variables and extracted financial data from CRSP with which we built models analysing the behaviour of the coefficient estimates. 

The code here convert the text of 3000 text documents of size 3.8GB each representing an IPO company into a single dataframe with each row representing with bigrams and trigrams of each document.

There are several stages involved in this task where we finally used elastic net regression model to predict the stck price. I am not presenting other steps of code due to confidentiality.
