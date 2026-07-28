LLM Ranking Regression Model Building and Evaluation

This dataset is pulled from Kaggle, and gives user rankings of AI models since inception.

The dataset is 93,377 rows with 12 columns.

Found a significant number of null values in the organization column, as well as 307 rows where the website where the data was collected from did not publish confidence intervals (leaving 4 columns blank for these 307 rows). In order to deal with these null values, I have imputed the categorical column nulls with the mode. I then imputed the numerical column nulls with the median if the distribution of the values were either right of left skewedness > 0.5, or the mean if the distribution of the values was closer to normal.

I then conducted an EDA using ProfileReport from the ydata_profiling library.
EDA findings:

1. Ranking, ranking_lower, and ranking_upper showed very strong correlations (0.996). 
ranking_upper and ranking_lower both dropped to reduce leakage as they are just the 95% confidence interval calculated from the ranking column (multicolinearity not a problem with RIDGE/LASSO, just with linear regression). 
2. While rank does not have a high correlation with rating across the entire dataset, when looking at leaderboard publish date grouped with the subset column there is in fact a high negative correlation, so I will drop rank as a feature due to potential leakage
3. Variance and vote count showed strong negative correlation, consistent with the expected statistical relationship where estimates based on more observations have lower variance — this reflects standard error behavior rather than data redundancy.

After the EDA, I moved on to modeling. Goal was to predict rating using organization, license, category, subset, vote_count, and leaderboard_publish_date as features. rating_lower, rating_upper, and variance got dropped since they're derived straight from rating, and rank got dropped too per the leakage finding above.

I trained 4 models to compare: Decision Tree, Random Forest, XGBoost, and Linear Regression (added this last one just as a baseline to see how much the tree models were actually buying me over a plain linear fit).

First run, using a normal train/test split, gave me these results:

Random Forest - MAE 4.65, RMSE 15.61, R2 0.988
XGBoost - MAE 27.16, RMSE 39.31, R2 0.927
Decision Tree - MAE 59.55, RMSE 81.08, R2 0.688
Linear Regression - MAE 95.05, RMSE 116.81, R2 0.353

That Random Forest R2 of 0.988 seemed too good, especially since a single Decision Tree was only getting 0.688. Turns out the issue was that this dataset has the same model_name showing up over and over across different leaderboard_publish_dates, so a normal row-level split was letting rows from the same model end up in both train and test. Random Forest was basically memorizing model_name-associated rows instead of learning anything generalizable.

Fixed this by switching to GroupShuffleSplit grouped on model_name so all rows for a given model stay entirely in train or entirely in test, no overlap. Reran everything and got more realistic numbers:

XGBoost - MAE 58.91, RMSE 75.53, R2 0.689
Random Forest - MAE 70.65, RMSE 94.38, R2 0.514
Decision Tree - MAE 77.37, RMSE 100.44, R2 0.449
Linear Regression - MAE 96.54, RMSE 118.06, R2 0.239

Once the leakage was fixed, the ranking made a lot more sense (XGBoost > Random Forest > Decision Tree > Linear Regression) and Random Forest's R2 dropped from 0.988 down to 0.514, confirming the first run was inflated.

Given that none of my features actually describe how good a model is (no benchmark scores, no parameter counts, nothing about the model itself really) getting an R2 of 0.689 out of XGBoost using just organization/license/subset/vote_count/date feels like a solid result. Linear Regression being way lower (0.239) tells me the relationship isn't linear, which is probably why the tree-based models do better here. If I wanted to push this further I'd want to add features that actually reflect capability instead of just tuning the models more.
