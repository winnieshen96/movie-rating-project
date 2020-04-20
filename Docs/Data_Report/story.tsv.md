## General summary of the data
The plot summaries or synopsis of 5246 movies. Only movies that meet the following two requirements will appear in this dataset:
* appear in budget.tsv
* its name in budget.tsv has corresponding movie id in title.basics.tsv
## Data quality summary
Completeness is okay, although there are over 600,000+ movies in the IMDB datasets, there are only 5246 movies with budget and gross data that can be used for modeling.
Luckily for those movies that satisfy the requirements, each of them has corresponding plot summaries scraped from IMDB. There are several things that need attention.
* When scraped, the program prioritizes synopsis over plot summaries, and the detailed definition of synopsis and plot summaries can be found [here](https://help.imdb.com/article/contribution/titles/plots/G56STCKTK7ESG7CP?ref_=helpsrall#)
* The punctuation marks are preserved.
## Target variable
None
## Individual variables
tconst, plot
## Variable ranking
Plots are both very important in terms of defining the success of a movie.
## Relationship between explanatory variables and target variable
The emotion arc of the plot can instigate viewer's emotions, hence having a relationship with viewer's experience. However, that relationship is yet to be determined. The project plans to fit a neural network with the plots as input and ROI as outputs.
