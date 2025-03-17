# Profit Prophet: The Stock Market ML Predictor

This project uses LSTM machine learning to make predictions on the stock market and recommend a course of action.

## Stock Data

TBD. What specific data should we collect for the LSTM?

## Approach

First we train an LSTM model for the [TBD. Aggreggated index? Each stock?] to get a forecast of the market based on pure numeric data.

Simultaneously, we read the news and use ML to catagorize the news and react in these ways:

|    Stock Implication   |          Past          |         Present           |          Future          |
| :--------------------: | :--------------------: | :-----------------------: | :----------------------: |
| Artificially Increased |  Reduce Past Estimate  |  Reduce Present Estimate  |  Reduce Future Estimate  |
| Artificially Decreased | Increase Past Estimate | Increase Present Estimate | Increase Future Estimate |
|        No Change       |       Do Nothing       |         Do Nothing        |        Do Nothing        |

This gives us an estimate and forecast of the *True* value of the stock, which we can use to make fat stacks.

## Tools

There are multiple different ways to get the stock prices; Bloomberg Terminals, and OpenBB

(More underfitting)
![image](https://github.com/user-attachments/assets/2af2493e-6d34-4803-9281-5657f0dc16a7)

(More overfitting)
![image](https://github.com/user-attachments/assets/0ab328da-cc07-4e80-aa7c-6c741ca1c5a9)
