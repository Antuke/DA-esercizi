library(corrplot) # libreria per correlation matrix plotting
library(leaps) # libreria per subset selection
library(pracma) # per nthroot
library(car) # per vif
library(glmnet)# per ridge e lasso

rm(list=ls())
graphics.off()
# Function to create a synthetic dataset with random polynomial terms
create_synthetic_dataset <- function(n = 160, max_degree = 10, include_prob = 0.5, noise_sd = 5) {
  X <- runif(n, min = -1, max = 1)  # X values uniformly distributed between -10 and 10
  
  # Randomly decide which polynomial terms to include
  terms <- 1:max_degree  # Consider terms from X^1 to X^max_degree
  included_terms <- terms[runif(length(terms)) < include_prob]  # Include terms with probability include_prob
  
  # If no terms are included, default to X^1 (linear term)
  if (length(included_terms) == 0) {
    included_terms <- c(1)
  }
  
  # Generate the dependent variable Y with the included polynomial terms
  Y <- rowSums(sapply(included_terms, function(term) X^term)) + rnorm(n, mean = 0, sd = noise_sd)
  
  # Combine X and Y into a data frame
  data <- data.frame(X = X, Y = Y)
  
  # Return the dataset and the included terms for reference
  return(list(data = data, included_terms = included_terms))
}

# Generate synthetic data
synthetic <- create_synthetic_dataset(n = 160, max_degree = 10, include_prob = 0.3, noise_sd = 1)
data <- synthetic$data
included_terms <- synthetic$included_terms

# Print the included terms
print("Included polynomial terms:")
print(included_terms)

# Define maximum polynomial degree to test
max_degree <- 10

# Cross-validation setup
k <- 5  # Number of folds
folds <- sample(rep(1:k, length.out = nrow(data)))  # Assign folds
cv_errors <- matrix(NA, nrow = k, ncol = max_degree)  # Store errors for each fold and degree

# Perform cross-validation
for (degree in 1:max_degree) {
  for (fold in 1:k) {
    # Split data into training and validation sets
    train_data <- data[folds != fold, ]
    test_data <- data[folds == fold, ]
    
    # Fit polynomial regression model
    model <- lm(Y ~ poly(X, degree, raw = TRUE), data = train_data)
    
    # Predict on the validation set
    predictions <- predict(model, newdata = test_data)
    
    # Calculate mean squared error
    cv_errors[fold, degree] <- mean((test_data$Y - predictions)^2)
  }
}

# Calculate mean cross-validation error for each degree
mean_cv_errors <- colMeans(cv_errors)

# Print results
results <- data.frame(Degree = 1:max_degree, MSE = mean_cv_errors)

# Plot MSE vs. polynomial degree
plot(results$Degree, results$MSE, type = "b", xlab = "Polynomial Degree", ylab = "Mean Squared Error (MSE)",
     main = "MSE vs. Polynomial Degree")
best_degree <- which.min(mean_cv_errors)
cat("Best degree ",best_degree,"\n")
abline(v = best_degree, col = "red", lty = 2) 


#----------------------#
new_data = data.frame(Y = data$Y)
for (i in 1:best_degree) {
  new_data[[paste0("X", i)]] <- data$X^i
}


regfit.bwd=regsubsets(Y~.,data=new_data,nvmax=best_degree, method ="backward");
summ.bwd = summary(regfit.bwd);

# stime delle perfomance dei modelli con bic e adjr2
best_bic_bwd_index = which.min(summ.bwd$bic); # -2LogL + log(n)d
best_adjustedR2_bwd_index = which.max(summ.bwd$adjr2);

# plot bic,adjr2, test_error (MSE)
dev.new();
par(mfrow=c(2,1), oma=c(0, 0, 2, 0));
plot(1:best_degree,summ.bwd$bic,xlab="Number of regressor",ylab="bic");
points(best_bic_bwd_index, summ.bwd$bic[best_bic_bwd_index], col = "red",pch=15);
plot(1:best_degree,summ.bwd$adjr2,xlab="Number of regressor",ylab="adjustedR2");
points(best_adjustedR2_bwd_index, summ.bwd$adjr2[best_adjustedR2_bwd_index], col = "red",pch=15);



regressor_ids = names(coef(regfit.bwd,id=best_bic_bwd_index))



mtext("STEPWISE BACKWARD", side=3, line=0, outer=TRUE, cex=1.5);
cat("\n----------------------STEPWISE-BACKWARD----------------------\n")
cat("Best bic value     =",summ.bwd$bic[best_bic_bwd_index],"\tNumber of regressor=",best_bic_bwd_index,"\n")
cat("Best adjustedR2    =",summ.bwd$adjr2[best_adjustedR2_bwd_index],"\tNumber of regressor=",best_adjustedR2_bwd_index,"\n")

cat("Id of regressors for best BIC =", regressor_ids)


final_model <- lm(Y ~ ., data = new_data[, c("Y", regressor_ids[-1])])
plot(final_model)

# ---------------- #



# First filter the data for Y values between -100 and 100
filtered_data <- new_data[new_data$Y >= -300 & new_data$Y <= 300, ]

# Create relevant_data from filtered data
relevant_data <- filtered_data[, c("Y", regressor_ids[-1])]

confint(final_model)
dev.new()

# Create ordered indices for X1
ordered_indices <- order(filtered_data$X1)
xx <- filtered_data$X1[ordered_indices]

# Plot the filtered and ordered data points
plot(filtered_data$X1, filtered_data$Y, 
     xlab = "X1", ylab = "Y", 
     pch = 16, col = "black",
     main = "Regression Function with Confidence and Prediction Intervals")

# Get predicted values for ordered data
ordered_relevant_data <- relevant_data[ordered_indices, ]
yy <- predict(final_model, newdata = ordered_relevant_data)

# Plot the regression line
lines(xx, yy, col = "red", lwd = 4)

# Add confidence intervals
ci_lin <- predict(final_model, newdata = ordered_relevant_data, interval = "confidence")
lines(xx, ci_lin[, "lwr"], lty = 2, lwd = 2, col = "blue")  # Lower confidence bound
lines(xx, ci_lin[, "upr"], lty = 2, lwd = 2, col = "blue")  # Upper confidence bound

# Add prediction intervals
pi_lin <- predict(final_model, newdata = ordered_relevant_data, interval = "prediction")
lines(xx, pi_lin[, "lwr"], lty = 2, lwd = 2, col = "green")  # Lower prediction bound
lines(xx, pi_lin[, "upr"], lty = 2, lwd = 2, col = "green")  # Upper prediction bound

# Add legend
legend('topright', 
       legend = c('Data', 'Regression Line', '0.95 Conf. Bound', '0.95 Pred. Bound'), 
       col = c('black', 'red', 'red', 'green'), 
       lty = c(NA, 1, 2, 2), 
       pch = c(16, NA, NA, NA), 
       lwd = c(NA, 4, 2, 2), 
       cex = 0.9)








