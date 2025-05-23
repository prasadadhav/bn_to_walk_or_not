# Experimenting with Bayesian neural networks (BNN).

In this repo I experiment with Bayesian neural network. The idea is to create a simple scenario, i.e predict if I would go for a walk. 


## Step 1: create dummy data
I create a dummy data based on some realistic temperature values for my location. I use `beautifulsoup` to scrape atable that shows temperature for day, night, and how many days it rains, for each month.


![Pairs Plot of Dummy Data](./images/pairs_plot_seasons.png)

This pair plot is created using `creatdummy_data.ipynb`.

## Step 2: Train BNN on the dummy data
In this step I create and train the BNN.
![Model 1 Bayesian Neural Network](./images/model_1_bn.png)

## Step 3: Do predictions, testing, & fine tuning
Here I create certain scenarios and test out what my BNN does.
Here I take the opportunity to explore the decision make through the nodes, and explore the network.
![Predictions of Walks Through the Months](./images/pred_walk_months.png)

In this step, I predict if I would go for a walk through the months.

## Step 4: ML & Data Ops
In this step I use github actions to add data everyday