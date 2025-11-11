First, we loaded and cleaned the training and test datasets, ensuring that all samples under attack were excluded to maintain data integrity.
Sensor readings were normalized to allow for fair comparison between datasets. 
We then computed the Kolmogorov–Smirnov (K–S) statistic for each sensor to quantify distribution differences between the train and test sets. 
To assess system state coverage, we identified all unique actuator states in both datasets and calculated the percentage of common states present in both sets, as well as the proportion of states in one set that are covered by the other. 
This approach provides insight into the representativeness and overlap of normal operating conditions between the training and test data,
for each common state we extract sensor values and compute n number of K-s statistic
from this we calculate the averate ks - statistic
next we take all sensor row attributed to common state and calulate their Ks statistic
At this stage we have ks statistic with states and without states. 
At the end we compute CCDFs of those two ks values. and plot them

we repeat this step for all three data sets.