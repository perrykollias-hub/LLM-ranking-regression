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
