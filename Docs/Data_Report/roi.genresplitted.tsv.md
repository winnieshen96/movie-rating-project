## General summary of the data
Calculated ROI for domestic and worldwide.
## Data quality summary
Completeness maintained, quality is good.
## Data Cleaning summary
For linear regression, we need to remove some outliers. However for machine learning classification problem, since we label all movies with ROI < 2 as 1 and other movies as 0, we eliminate the problem of value ranges, therefore we don't need to remove movies with exceptionally high ROIs.
Before removing outliers (ROI > 80):
![alt text](https://github.com/winnieshen96/movie-rating-project/blob/master/Docs/Images/roi.boxplot.before.png)

After removing outliers:
![alt text](https://github.com/winnieshen96/movie-rating-project/blob/master/Docs/Images/roi.boxplot.after.png)
