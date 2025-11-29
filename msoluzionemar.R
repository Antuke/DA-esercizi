rm(list=ls())
graphics.off()

library(corrplot) # libreria per correlation matrix plotting
library(leaps) # libreria per subset selection
library(pracma) # per nthroot
library(car) # per vif
library(glmnet)# per ridge e lasso

# genera un test set sintetico, di cui il prob_uselss di variabili indipendenti
# sono inutili, ovvero hanno il beta associato pari a 0 
generate_synthetic_data <- function(n, p, prob_useless = 0.5) {
  # Set seed for reproducibility
  #set.seed(123)
  
  # Generate the regressors X (n x p matrix)
  X <- matrix(rnorm(n * p), nrow = n, ncol = p)
  
  # Assign names to the regressors (X1, X2, ..., Xp)
  colnames(X) <- paste0("X", 1:p)
  
  # Generate true coefficients (betas)
  # Each beta has a probability `prob_useless` of being 0 (useless)
  true_betas <- ifelse(runif(p) < prob_useless, 0, rnorm(p))
  
  # Assign names to the coefficients (beta1, beta2, ..., betap)
  names(true_betas) <- paste0("X", 1:p)
  
  # Generate the intercept (beta_0)
  beta_0 <- rnorm(1)  # Intercept is drawn from a standard normal distribution
  names(beta_0) <- "(Intercept)"  # Name the intercept
  
  # Combine the intercept and the coefficients into a single named vector
  all_coefficients <- c(beta_0, true_betas)
  
  # Generate the dependent variable Y
  epsilon <- rnorm(n, mean = 0, sd = 0.5)  # Noise term
  Y <- beta_0 + X %*% true_betas + epsilon
  
  # Combine X and Y into a single dataframe
  data <- data.frame(Y = Y, X)
  
  # Return the dataframe and all coefficients (including intercept) in one vector
  return(list(data = data, coefficients = all_coefficients))
}
n = 50
p = 22
synthetic = generate_synthetic_data(n = 500, p = 22)
#data = read.csv("./RegressionDA2401102024.csv");
#betas = synthetic$coefficients

# split in in training e test set
split = sample(nrow(synthetic$data) , nrow(synthetic$data) * 0.7);

data = synthetic$data
data_train = synthetic$data[split,]
data_test = synthetic$data[-split,]


# Check preliminari (ricerca di correlazione evidente nei dati)
dev.new();
coll_matrix = cor(data[,2:ncol(data)]);
corrplot(coll_matrix, method="number");

dev.new()
plot(data_train)

fit = lm(Y ~ .,data_train);
summary(fit);
test_loss = mean( ( data_test[,1] - predict(fit,data_test[,2:(p+1)]) )^2 )
cat("\n----------------------FULL-LINEAR-30-REGRESSOR----------------------\n")
cat("test-set loss:",test_loss ) 
vif(fit)

# no correlazione (i dati sono generati casaulmente con la distribuzione normale)


# --------------------- EXHAUSTIVE ----------------------------- #
regfit.full=regsubsets(Y~.,data=data_train,nvmax=p, method ="exhaustive");
summ.full = summary(regfit.full);

# stime delle perfomance dei modelli con bic e adjr2
best_bic_full_index = which.min(summ.full$bic); # -2LogL + log(n)d
best_adjustedR2_full_index = which.max(summ.full$adjr2);

# plot bic,adjr2, test_error (MSE)
dev.new();
par(mfrow=c(4,1), oma=c(0, 0, 2, 0));
plot(1:p,summ.full$bic,xlab="Number of regressor",ylab="bic");
points(best_bic_full_index, summ.full$bic[best_bic_full_index], col = "red",pch=15);
plot(1:p,summ.full$adjr2,xlab="Number of regressor",ylab="adjustedR2");
points(best_adjustedR2_full_index, summ.full$adjr2[best_adjustedR2_full_index], col = "red",pch=15);

# on test set
test_x = data_test[,2:ncol(data_test)]
test_y = data_test[,1]
train_x = data_train[,2:ncol(data_train)] 
train_y = data_train[,1]
te = numeric(p);
for(i in 1:p){
  b_i = coef(regfit.full,i);
  predictor_id = names(b_i)[-1] ;
  test_xi = test_x[,predictor_id];
  test_xi = cbind(1,test_xi);
  preds_i = as.matrix(test_xi) %*% b_i;
  te[i] = mean((preds_i - test_y)^2);
}
plot(1:p,te,xlab="Number of regressor",ylab="MSE_TEST_SET");
points(which.min(te),te[which.min(te)],col = "red",pch=15);





# cross-validation
k = 12;
folds = sample(1:k, nrow(data_train), replace=TRUE); # assign k-fold to each sample
cv.errors = matrix(nrow=k, ncol=p);  # Fixed variable name from cv.errors_full to cv.errors

for(j in 1:k){
  # all the data that are not in the j-fold are used for training
  train_subset = data_train[folds != j,]; # folds != j creates a mask
  best.fit = regsubsets(Y~., data=train_subset, nvmax=p, method="exhaustive");
  
  # validation data is the j fold
  cvtest_x = train_x[folds == j,];
  cvtest_y = train_y[folds == j];
  
  for(i in 1:p){
    
    model_coef = coef(best.fit, id=i);
    
    design_matrix = model.matrix(~., data=cvtest_x[, names(model_coef)[-1], drop=FALSE]);
    
    pred = design_matrix %*% model_coef;
    
    # Calculate mean squared error
    cv.errors[j,i] = mean((cvtest_y - pred)^2);
  }
}

# mean cross-validation error, ogni riga è un fold diverso
cv.errors = na.omit(cv.errors)
mean.cv.errors = colMeans(cv.errors);
plot(1:p, mean.cv.errors, xlab="Number of regressors", ylab="Cross-Validation Error");
points(which.min(mean.cv.errors),mean.cv.errors[which.min(mean.cv.errors)],col = "red",pch=15);

cat("\n----------------------STEPWISE-EXHAUSTIVE----------------------\n")
cat("Best bic value     =",summ.full$bic[best_bic_full_index],"\tNumber of regressor=",best_bic_full_index,"\n")
cat("Best test-set loss =",te[which.min(te)],"\tNumber of regressor=",which.min(te),"\n")
cat("Best adjustedR2    =",summ.full$adjr2[best_adjustedR2_full_index],"\tNumber of regressor=",best_adjustedR2_full_index,"\n")
cat("Best cv-loss k=12  =",mean.cv.errors[which.min(mean.cv.errors)],"\tNumber of regressor=",which.min(mean.cv.errors),"\n")


regressor_ids_cv_ex = names(coef(regfit.full,id=which.min(mean.cv.errors)))
regressor_ids_bic_ex =  names(coef(regfit.full,id=best_bic_full_index))

b_i = coef(regfit.full,best_bic_full_index);
predictor_id = names(b_i)[-1] ;
test_xi = test_x[,predictor_id];
test_xi = cbind(1,test_xi);
preds_i = as.matrix(test_xi) %*% b_i;
best_bic_test_loss = mean((preds_i - test_y)^2);


# --------------------- RIDGE-REGRESSION ----------------------------- #
test_x = data_test[,2:ncol(data_test)]
test_y = data_test[,1]
train_x = data_train[,2:ncol(data_train)] 
train_y = data_train[,1]

ridge = cv.glmnet(as.matrix(train_x),as.matrix(train_y),alpha = 0,nfolds=12)
dev.new()
plot(ridge)

preds = predict(ridge, as.matrix(test_x) , s = "lambda.min")
mse_ridge = mean( (test_y - preds)^2 )

cv_loss_best_lambda_ridge = ridge$cvm[which(ridge$lambda == ridge$lambda.min)]

cat("\n----------------------RIDGE-REGRESSION----------------------\n")
cat("Lambda-value  = ",ridge$lambda.min, "Lambda 1-se =",ridge$lambda.1se,"\n")
cat("Test-set loss = ", mse_ridge,"\n")
cat("Cv-loss lambda 1-se = ",cv_loss_best_lambda_ridge)

# --------------------- LASSO-REGRESSION ----------------------------- #

lasso = cv.glmnet(as.matrix(train_x),as.matrix(train_y),alpha = 1,nfolds=12)

dev.new()
plot(lasso)

cv_loss_best_lambda = lasso$cvm[which(lasso$lambda == lasso$lambda.1se)]

coef_min = as.matrix(coef(lasso,s=lasso$lambda.1se))
num_regressor = sum(coef_min != 0) - 1 # meno intercetta
regressor_ids_lasso = which(coef_min != 0) - 1

preds = predict(lasso, as.matrix(test_x) , s = "lambda.1se")

lasso_mse = mean( (test_y - preds)^2 )
cat("\n----------------------LASSO-REGRESSION----------------------\n")
cat("Lambda-value          = ",lasso$lambda.min, "Lambda 1-se =",lasso$lambda.1se,"\n")
cat("Number of regressors  = ",num_regressor,"\n")
cat("Id of regressors      = ", regressor_ids_lasso,"\n")
cat("Test-set loss         = ",lasso_mse,"\n")
cat("Cv-loss lambda 1-se   = ",cv_loss_best_lambda)


# --------------------- CONSIDERATIONS ----------------------------- #

cat("\n----------------------FINAL-CONSIDERATION----------------------\n")
cat("Id of regressors for best bic exhaustive =", regressor_ids_cv_ex)
cat("\nTest-MSE exhaustive (best-bic) =", best_bic_test_loss)
cat("\nTest-MSE ridge       =", cv_loss_best_lambda_ridge)
cat("\nId of regressors for best test-set loss lasso=", regressor_ids_lasso)
cat("\nTest-MSE lasso       =", lasso_mse)
#cat("\n\nTrue betas  =", names(which(betas!=0)))
#cat("\nTrue betas values = ", betas[betas!=0])
cat("\nBetas exhaustive =", coef(regfit.full,id=best_bic_full_index))

