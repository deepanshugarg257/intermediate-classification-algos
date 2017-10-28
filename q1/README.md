Kernel Trick
In this, i just tried the kernel mentioned in the report, and it worked like a charm.

Letter Classification
> I considered three kernels for SVM namely - poly, rbf and linear
> The hyperparameters used were C and gamma. Both were ranged from 0.1 to 100 and best one was chosen.
> For poly, degrees of 2,3,4,5 were considered.
> In the final submission, combination of (kernel='rbf', C=10, gamma=100) is used since it gave best accuracy of 97.06%.
> The dataset is split in five different folds and individually trained and tested on a part of the split. Finally, mean of accuracy, precision, recall and F1 score is taken across all the 5 splits. 
