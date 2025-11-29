library(corrplot) # libreria per correlation matrix plotting
library(leaps) # libreria per subset selection
library(pracma) # per nthroot
library(car) # per vif
library(glmnet)# per ridge e lasso
#set.seed(42)
rm(list=ls())
graphics.off()
# --------------------- DATASET ----------------------------- #
# lettura del datataset
data = read.csv("./data/Regression2024.csv");
#data[,1] = nthroot(data[,1],3)
n = nrow(data);
p = ncol(data) - 1;


# split del dataset in 80% train, 20 % test
split = sample(n , n * 0.8);

# prendi solo le righe che corrispondono con split = 1
data_train = data[split,];
#data_train[,1] = nthroot(data_train[,1],3)
#data_train[,1] = scale(data_train[,1])

#center = attr(data_train[,1],"scaled:center") 
#scale  = attr(data_train[,1],"scaled:scale")

#descale = function(data){
#  return ((data*center)-scale))
#}

data_test = data[-split,] ;

train_x = data_train[,2:ncol(data)];
train_y = data_train[,1];

test_x = data_test[, 2:ncol(data)];
test_y = data_test[, 1];

n_split = nrow(train_x);


# --------------------- CORRELATION MATRIX  ----------------------------- #
dev.new();
coll_matrix = cor(data[,2:ncol(data)]);
corrplot(coll_matrix, method="number");

#pairs(Y~X1+X2+X3+X4,data = data_train)
dev.new()
plot(data_train)
# il massimo valore di correlazione è tra le variabili x29 e x7 pari a 0.39
# che è un valore relativamente basso


# --------------------- MANUAL BETA ESTIMATION  ----------------------------- #
# aggiungiamo colonna di uno a train_x
X = as.matrix(cbind(1,train_x));
Y = train_y;
B <- solve(t(X)%*%X, t(X)%*%Y);


# --------------------- CALCOLO P-VALUES  ----------------------------- #
## calcolo p-value CDF_T_DISTRIBUTION(|t| > t_statistics) = p_value
# t_statistics = b/se
preds = X %*% B;
dof = n_split-p-1;
varY = sum((preds - Y)^2) / dof; # unbiased estimate di varY
SE = sqrt(diag(varY * solve(t(X) %*% X)));
t_statistics = B / SE;
p_values = 2 * pt(abs(t_statistics), df = dof, lower.tail = FALSE);
mask = p_values > 0.05;
B_p = B[mask];



# --------------------- LINEAR REGRESSION CON LM  ----------------------------- #
# I risultati combaciano (meno approssimazione per valori vicinissimi a 0)
fit = lm(Y ~ .,data_train);
summary(fit);
p_values_fit = summary(fit)$coefficients[,4];
test_loss = mean( ( test_y - predict(fit,test_x) )^2 )
cat("\n----------------------FULL-LINEAR-30-REGRESSOR----------------------\n")
cat("test-set loss:",test_loss ) 
vif(fit) # anche la vif mostra che non c'è una forte correlazione tra i regressori
         # vif > 5 può essere problematica, 10 quasi sicuro c'è multicollinearità
# --------------------- STEPWISE-FORWARD----------------------------- #
regfit.fwd=regsubsets(Y~.,data=data_train,nvmax=p, method ="forward");
summ.fwd = summary(regfit.fwd);

# stime delle perfomance dei modelli con bic e adjr2
best_bic_fwd_index = which.min(summ.fwd$bic); # -2LogL + log(n)d
best_adjustedR2_fwd_index = which.max(summ.fwd$adjr2);

# plot bic,adjr2, test_error (MSE)
dev.new();
par(mfrow=c(4,1),oma=c(0, 0, 2, 0));
plot(1:30,summ.fwd$bic,xlab="Number of regressor",ylab="bic");
points(best_bic_fwd_index, summ.fwd$bic[best_bic_fwd_index], col = "red",pch=15);
plot(1:30,summ.fwd$adjr2,xlab="Number of regressor",ylab="adjustedR2");
points(best_adjustedR2_fwd_index, summ.fwd$adjr2[best_adjustedR2_fwd_index], col = "red",pch=15);

te = numeric(30);
for(i in 1:30){
  b_i = coef(regfit.fwd,i);
  predictor_id = names(b_i)[-1] ;
  test_xi = test_x[,predictor_id];
  test_xi = cbind(1,test_xi);
  preds_i = as.matrix(test_xi) %*% b_i;
  te[i] = mean((preds_i - test_y)^2);
}

plot(1:30,te,xlab="Number of regressor",ylab="MSE");
points(which.min(te),min(te),col = "red",pch=15);
mtext("STEPWISE FORWARD", side=3, line=0, outer=TRUE, cex=1.5);


# cross-validation
k = 12;
folds = sample(1:k, nrow(data_train), replace=TRUE); # assign k-fold to each sample
cv.errors = matrix(nrow=k, ncol=p);  # Fixed variable name from cv.errors_full to cv.errors

for(j in 1:k){
  # all the data that are not in the j-fold are used for training
  train_subset = data_train[folds != j,]; # folds != j creates a mask
  best.fit = regsubsets(Y~., data=train_subset, nvmax=p, method="forward");
  
  # validation data is the j fold
  cvtest_x = train_x[folds == j,];
  cvtest_y = train_y[folds == j];
  
  for(i in 1:30){
    
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
plot(1:30, mean.cv.errors, xlab="Number of regressors", ylab="Cross-Validation Error");
points(which.min(mean.cv.errors),mean.cv.errors[which.min(mean.cv.errors)],col = "red",pch=15);




cat("\n----------------------STEPWISE-FORWARD----------------------\n")
cat("Best bic value     =",summ.fwd$bic[best_bic_fwd_index],"\tNumber of regressor=",best_bic_fwd_index,"\n")
cat("Best test-set loss =",te[which.min(te)],"\tNumber of regressor=",which.min(te),"\n")
cat("Best adjustedR2    =",summ.fwd$adjr2[best_adjustedR2_fwd_index],"\tNumber of regressor=",best_adjustedR2_fwd_index,"\n")
cat("Best cv-loss k=12  =",mean.cv.errors[which.min(mean.cv.errors)],"\tNumber of regressor=",which.min(mean.cv.errors),"\n")


regressor_ids = names(coef(regfit.fwd,id=which.min(mean.cv.errors)))
cat("Id of regressors for best cv =", regressor_ids)


# --------------------- STEPWISE-BACKWARD----------------------------- #
regfit.bwd=regsubsets(Y~.,data=data_train,nvmax=30, method ="backward");
summ.bwd = summary(regfit.bwd);

# stime delle perfomance dei modelli con bic e adjr2
best_bic_bwd_index = which.min(summ.bwd$bic); # -2LogL + log(n)d
best_adjustedR2_bwd_index = which.max(summ.bwd$adjr2);

# plot bic,adjr2, test_error (MSE)
dev.new();
par(mfrow=c(4,1), oma=c(0, 0, 2, 0));
plot(1:30,summ.bwd$bic,xlab="Number of regressor",ylab="bic");
points(best_bic_bwd_index, summ.bwd$bic[best_bic_bwd_index], col = "red",pch=15);
plot(1:30,summ.bwd$adjr2,xlab="Number of regressor",ylab="adjustedR2");
points(best_adjustedR2_bwd_index, summ.bwd$adjr2[best_adjustedR2_bwd_index], col = "red",pch=15);

te = numeric(30);
for(i in 1:30){
  b_i = coef(regfit.bwd,i);
  predictor_id = names(b_i)[-1] ;
  test_xi = test_x[,predictor_id];
  test_xi = cbind(1,test_xi);
  preds_i = as.matrix(test_xi) %*% b_i;
  te[i] = mean((preds_i - test_y)^2);
}

plot(1:30,te,xlab="Number of regressor",ylab="MSE");
points(which.min(te),te[which.min(te)],col = "red",pch=15);


# cross-validation
k = 12;
#folds = sample(1:k, nrow(data_train), replace=TRUE); # assign k-fold to each sample
cv.errors = matrix(nrow=k, ncol=p);  # Fixed variable name from cv.errors_full to cv.errors

for(j in 1:k){
  # all the data that are not in the j-fold are used for training
  train_subset = data_train[folds != j,]; # folds != j creates a mask
  best.fit = regsubsets(Y~., data=train_subset, nvmax=p, method="backward");
  
  # validation data is the j fold
  cvtest_x = train_x[folds == j,];
  cvtest_y = train_y[folds == j];
  
  for(i in 1:30){
    
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
plot(1:30, mean.cv.errors, xlab="Number of regressors", ylab="Cross-Validation Error");
points(which.min(mean.cv.errors),mean.cv.errors[which.min(mean.cv.errors)],col = "red",pch=15);




mtext("STEPWISE BACKWARD", side=3, line=0, outer=TRUE, cex=1.5);
cat("\n----------------------STEPWISE-BACKWARD----------------------\n")
cat("Best bic value     =",summ.bwd$bic[best_bic_bwd_index],"\tNumber of regressor=",best_bic_bwd_index,"\n")
cat("Best test-set loss =",te[which.min(te)],"\tNumber of regressor=",which.min(te),"\n")
cat("Best adjustedR2    =",summ.bwd$adjr2[best_adjustedR2_bwd_index],"\tNumber of regressor=",best_adjustedR2_bwd_index,"\n")
cat("Best cv-loss k=12  =",mean.cv.errors[which.min(mean.cv.errors)],"\tNumber of regressor=",which.min(mean.cv.errors),"\n")

regressor_ids = names(coef(regfit.bwd,id=which.min(mean.cv.errors)))
cat("Id of regressors for best cv =", regressor_ids)



# --------------------- STEPWISE-HYBRID----------------------------- #
regfit.hyb=regsubsets(Y~.,data=data_train,nvmax=30, method ="seqrep");
summ.hyb = summary(regfit.hyb);

# stime delle perfomance dei modelli con bic e adjr2
best_bic_hyb_index = which.min(summ.hyb$bic); # -2LogL + log(n)d
best_adjustedR2_hyb_index = which.max(summ.hyb$adjr2);

# plot bic,adjr2, test_error (MSE)
dev.new();
par(mfrow=c(4,1), oma=c(0, 0, 2, 0));
plot(1:30,summ.hyb$bic,xlab="Number of regressor",ylab="bic");
points(best_bic_hyb_index, summ.hyb$bic[best_bic_hyb_index], col = "red",pch=15);
plot(1:30,summ.hyb$adjr2,xlab="Number of regressor",ylab="adjustedR2");
points(best_adjustedR2_hyb_index, summ.hyb$adjr2[best_adjustedR2_hyb_index], col = "red",pch=15);

te = numeric(30);
for(i in 1:30){
  b_i = coef(regfit.hyb,i);
  predictor_id = names(b_i)[-1] ;
  test_xi = test_x[,predictor_id];
  test_xi = cbind(1,test_xi);
  preds_i = as.matrix(test_xi) %*% b_i;
  te[i] = mean((preds_i - test_y)^2);
}

plot(1:30,te,xlab="Number of regressor",ylab="MSE");
points(which.min(te),te[which.min(te)],col = "red",pch=15);

mtext("STEPWISE HYBRID", side=3, line=0, outer=TRUE, cex=1.5);

# cross-validation
k = 12;
#folds = sample(1:k, nrow(data_train), replace=TRUE); # assign k-fold to each sample
cv.errors = matrix(nrow=k, ncol=p);  # Fixed variable name from cv.errors_full to cv.errors

for(j in 1:k){
  # all the data that are not in the j-fold are used for training
  train_subset = data_train[folds != j,]; # folds != j creates a mask
  best.fit = regsubsets(Y~., data=train_subset, nvmax=p, method="seqrep");
  
  # validation data is the j fold
  cvtest_x = train_x[folds == j,];
  cvtest_y = train_y[folds == j];
  
  for(i in 1:30){
    
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
plot(1:30, mean.cv.errors, xlab="Number of regressors", ylab="Cross-Validation Error");
points(which.min(mean.cv.errors),mean.cv.errors[which.min(mean.cv.errors)],col = "red",pch=15);



cat("\n----------------------STEPWISE-HYBRID----------------------\n")
cat("Best bic value     =",summ.hyb$bic[best_bic_hyb_index],"\tNumber of regressor=",best_bic_hyb_index,"\n")
cat("Best test-set loss =",te[which.min(te)],"\tNumber of regressor=",which.min(te),"\n")
cat("Best adjustedR2    =",summ.hyb$adjr2[best_adjustedR2_hyb_index],"\tNumber of regressor=",best_adjustedR2_hyb_index,"\n")
cat("Best cv-loss k=12  =",mean.cv.errors[which.min(mean.cv.errors)],"\tNumber of regressor=",which.min(mean.cv.errors),"\n")



regressor_ids = names(coef(regfit.hyb,id=which.min(mean.cv.errors)))
cat("Id of regressors for best cv =", regressor_ids)






# --------------------- EXHAUSTIVE ----------------------------- #
regfit.full=regsubsets(Y~.,data=data_train,nvmax=30, method ="exhaustive");
summ.full = summary(regfit.full);

# stime delle perfomance dei modelli con bic e adjr2
best_bic_full_index = which.min(summ.full$bic); # -2LogL + log(n)d
best_adjustedR2_full_index = which.max(summ.full$adjr2);

# plot bic,adjr2, test_error (MSE)
dev.new();
par(mfrow=c(4,1), oma=c(0, 0, 2, 0));
plot(1:30,summ.full$bic,xlab="Number of regressor",ylab="bic");
points(best_bic_full_index, summ.full$bic[best_bic_full_index], col = "red",pch=15);
plot(1:30,summ.full$adjr2,xlab="Number of regressor",ylab="adjustedR2");
points(best_adjustedR2_full_index, summ.full$adjr2[best_adjustedR2_full_index], col = "red",pch=15);

# on test set
te = numeric(30);
for(i in 1:30){
  b_i = coef(regfit.full,i);
  predictor_id = names(b_i)[-1] ;
  test_xi = test_x[,predictor_id];
  test_xi = cbind(1,test_xi);
  preds_i = as.matrix(test_xi) %*% b_i;
  te[i] = mean((preds_i - test_y)^2);
}
plot(1:30,te,xlab="Number of regressor",ylab="MSE");
points(which.min(te),te[which.min(te)],col = "red",pch=15);





# cross-validation
k = 12;
#folds = sample(1:k, nrow(data_train), replace=TRUE); # assign k-fold to each sample
cv.errors = matrix(nrow=k, ncol=p);  # Fixed variable name from cv.errors_full to cv.errors

for(j in 1:k){
  # all the data that are not in the j-fold are used for training
  train_subset = data_train[folds != j,]; # folds != j creates a mask
  best.fit = regsubsets(Y~., data=train_subset, nvmax=p, method="exhaustive");
  
  # validation data is the j fold
  cvtest_x = train_x[folds == j,];
  cvtest_y = train_y[folds == j];
  
  for(i in 1:30){
    
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
plot(1:30, mean.cv.errors, xlab="Number of regressors", ylab="Cross-Validation Error");
points(which.min(mean.cv.errors),mean.cv.errors[which.min(mean.cv.errors)],col = "red",pch=15);

cat("\n----------------------STEPWISE-EXHAUSTIVE----------------------\n")
cat("Best bic value     =",summ.full$bic[best_bic_full_index],"\tNumber of regressor=",best_bic_full_index,"\n")
cat("Best test-set loss =",te[which.min(te)],"\tNumber of regressor=",which.min(te),"\n")
cat("Best adjustedR2    =",summ.full$adjr2[best_adjustedR2_full_index],"\tNumber of regressor=",best_adjustedR2_full_index,"\n")
cat("Best cv-loss k=12  =",mean.cv.errors[which.min(mean.cv.errors)],"\tNumber of regressor=",which.min(mean.cv.errors),"\n")


regressor_ids = names(coef(regfit.full,id=which.min(mean.cv.errors)))

cat("Id of regressors for best cv =", regressor_ids)


mtext("EXHAUSTIVE", side=3, line=0, outer=TRUE, cex=1.5);




# --------------------- RIDGE-REGRESSION ----------------------------- #

ridge = cv.glmnet(as.matrix(train_x),as.matrix(train_y),alpha = 0,nfolds=12)
dev.new()
plot(ridge)

preds = predict(ridge, as.matrix(test_x) , s = "lambda.min")
mse_ridge = mean( (test_y - preds)^2 )

cv_loss_best_lambda = ridge$cvm[which(ridge$lambda == ridge$lambda.min)]

cat("\n----------------------RIDGE-REGRESSION----------------------\n")
cat("Lambda-value  = ",ridge$lambda.min, "Lambda 1-se =",ridge$lambda.1se,"\n")
cat("Test-set loss = ", mse_ridge,"\n")
cat("Cv-loss lambda 1-se = ",cv_loss_best_lambda)

# --------------------- LASSO-REGRESSION ----------------------------- #

lasso = cv.glmnet(as.matrix(train_x),as.matrix(train_y),alpha = 1,nfolds=12)

dev.new()
plot(lasso)

cv_loss_best_lambda = lasso$cvm[which(lasso$lambda == lasso$lambda.1se)]

coef_min = as.matrix(coef(lasso,s=lasso$lambda.1se))
num_regressor = sum(coef_min != 0) - 1 # meno intercetta
regressor_ids = which(coef_min != 0) - 1

preds = predict(lasso, as.matrix(test_x) , s = "lambda.1se")

lasso_mse = mean( (test_y - preds)^2 )
cat("\n----------------------LASSO-REGRESSION----------------------\n")
cat("Lambda-value          = ",lasso$lambda.min, "Lambda 1-se =",lasso$lambda.1se,"\n")
cat("Number of regressors  = ",num_regressor,"\n")
cat("Id of regressors      = ", regressor_ids,"\n")
cat("Test-set loss         = ",lasso_mse,"\n")
cat("Cv-loss lambda 1-se   = ",cv_loss_best_lambda)

lasso = glmnet(as.matrix(train_x),as.matrix(train_y),alpha = 1)

#b_i = coef(regfit.full,14)
#predictor_id = names(b_i)[-1] 
#test_xi = test_x[,predictor_id]
#test_xi = cbind(1,test_xi)
#preds_i = as.matrix(test_xi) %*% b_i
#err_ex = mean((preds_i - test_y)^2)