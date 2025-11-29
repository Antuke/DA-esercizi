library(corrplot)
library(leaps)
library(pracma)
library(car)
library(glmnet)
rm(list=ls())
graphics.off()


# Function to create a synthetic dataset with random polynomial terms
create_synthetic_dataset <- function(n = 505, max_degree = 10, include_prob = 0.4, noise_sd = 0.5) {
  X <- runif(n, min = -2, max = 2)  # X values uniformly distributed between -10 and 10
  
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

max_degree = 15
synthetic = create_synthetic_dataset()
data = synthetic$data

split = sample(nrow(synthetic$data) , nrow(synthetic$data) * 0.8);

data = synthetic$data
data_train = synthetic$data[split,]
data_test = synthetic$data[-split,]


te = numeric(max_degree)

for (degree in 1:max_degree) {
  model <- lm(Y ~ poly(X, degree, raw = TRUE), data = data_train)
  
  preds <- predict(model, newdata = data.frame(X = data_test$X))
  
  te[degree] <- mean((preds - data_test$Y)^2)
}


cat("\nTruth : ", synthetic$included_terms)
cat("\nBest MSE from test-set: ",min(te)," obtained with ", which.min(te), "degrees")

best_degree = which.min(te)

data_train_expanded <- data.frame(Y = data$Y, poly(data$X, degree = best_degree, raw = TRUE))

regfit.fwd=regsubsets(Y~.,data=data_train_expanded,nvmax=best_degree, method ="forward")

summ.fwd = summary(regfit.fwd)

best_bic_index = which.min(summ.fwd$bic)
best_bic_value = min(summ.fwd$bic)

dev.new()
plot(1:best_degree,summ.fwd$bic,xlab="Number of regressor",ylab="bic")

cat("\nWith stepwise forward the regressor found where:\n",names(coef(regfit.fwd,id=best_bic_index)))

coef_name = names(coef(regfit.fwd,id=best_bic_index))[-1]
best_formula =  as.formula(paste("Y ~", paste(coef_name, collapse = " + ")))

model = lm(best_formula, data = data_train_expanded)
dev.new()
par(mfrow = c(2, 2))  # Set up the plotting area for a 2x2 grid of plots
plot(model)


# Generate a sequence of X values for plotting
x_seq <- seq(min(data$X), max(data$X), length.out = 100)

# Create a data frame for prediction
new_data <- data.frame(X = x_seq)
new_data_expanded <- data.frame(poly(new_data$X, degree = best_degree, raw = TRUE))
colnames(new_data_expanded) <- paste0("X", 1:best_degree)  # Ensure column names match the model


beta_interval = confint(model)
# Predict Y values for the model
predictions <- predict(model, newdata = new_data_expanded, interval = "confidence")
predictions_pred <- predict(model, newdata = new_data_expanded, interval = "prediction")

# Plotting
dev.new()
plot(data$X, data$Y, main = "Polynomial Regression with Confidence and Prediction Intervals", xlab = "X", ylab = "Y")
lines(x_seq, predictions[, "fit"], col = "blue", lwd = 2)
matlines(x_seq, predictions[, c("lwr", "upr")], col = "red", lty = 2)
matlines(x_seq, predictions_pred[, c("lwr", "upr")], col = "green", lty = 3)



legend("topright", legend = c("Fitted", "Confidence Interval", "Prediction Interval"), col = c("blue", "red", "green"), lty = c(1, 2, 3), lwd = c(2, 1, 1))








