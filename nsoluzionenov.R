# Generate dataset ----
n= 100
num_regressions= 30

independent_vars= matrix(rnorm(n * num_regressions, mean = 5, sd = 2), nrow=n, ncol=num_regressions)
colnames(independent_vars)= paste0("X", 1:num_regressions)

coefficients= runif(num_regressions, min = -2, max = 2)  # Random coefficients between -2 and 2

epsilon= rnorm(n, mean = 0, sd = 1)

y= 3 + independent_vars %*% coefficients + epsilon

dataset= data.frame(Y=as.vector(y), independent_vars)

names(dataset)

# Generate data ----
generate_synthetic_data <- function(n, p, prob_useless = 0.5) {
  # Set seed for reproducibility
  #set.seed(123)
  
  # Generate the regressors X (n x p matrix)
  X <- matrix(rnorm(n * p), nrow = n, ncol = p)
  
  # Assign names to the regressors (X1, X2, ..., Xp)
  colnames(X) <- paste0("X", 1:p)
  
  # Generate true coefficients (betas)
  # Each beta has a probability prob_useless of being 0 (useless)
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

result= generate_synthetic_data(100, 30)
dataset= result$data
cat(names(result$coefficients[which(result$coefficients!=0)]))

# Disione train e test ----
train_index= sample(1:nrow(dataset), size=0.8*nrow(dataset)) # size: numero di elementi da scegliere
test_index= -train_index

train_set= dataset[train_index, ]
test_set= dataset[-train_index, ]

# Forward method ----
library(leaps)
nvmax_value= dim(dataset)[2]-1

# La funzione serve per la predizione (non esiste già implementata)
# È scritta come 'predict.regsubsets' così è possibile chiamare direttamente solo
# 'predict' e funziona anche sull'oggetto di tipo 'regsubsets' senza necessità
# di dover chiamare una "funzione specifica"
# 
# Parametri:
#     object  --> Oggetto "lm.subsets" su cui si vuole effettuare la predizione
#     newdata --> Singolo valore oppure matrice di valori su cui si vuole calcolare la predizione
#     id      --> Indice del subset su cui effettuare il calcolo
# Restituisce:
#     Il valore predetto oppure una lista di valori predetti
predict.regsubsets= function(object, newdata, id, ...){
  form=   as.formula(object$call[[2]])  # ottenimento della formula utilizzata (es.: y~.)
  mat=    model.matrix(form, newdata)   # generazione della matrice 'X' di design
  coef_i= coef(object, id=id)           # estrazione dei coefficienti relativi all'indice passato
  xvars=  names(coef_i)                 # estrazione dei nomi delle colonne con cui moltiplicare i coefficienti
  mat[,xvars] %*% coef_i                # operazione di moltiplicazione => valori predetti
}

show_info= function(object){
  # L'ogetto passato deve essere del tipo 'summary(fit)'
  dev.new()
  par(mfrow=c(2,2))
  
  # Se si vuole visualizzare meglio si può usare << ylim=c(min, max) >>
  
  plot(object$rss, type='l', lwd='1', xlab="n° variables", ylab="RSS")
  points(which.min(object$rss), min(object$rss), type='p', col="red")
  
  plot(object$adjr2, type='l', lwd='1', xlab="n° variables", ylab="adjusted R2")
  points(which.max(object$adjr2), max(object$adjr2), type='p', col="red")
  
  plot(object$cp, type='l', lwd='1', xlab="n° variables", ylab="Cp")
  points(which.min(object$cp), min(object$cp), type='p', col="red")
  
  plot(object$bic, type='l', lwd='1', xlab="n° variables", ylab="BIC")
  points(which.min(object$bic), min(object$bic), type='p', col="red")  
}


fit.fwd= regsubsets(Y~., data=train_set, method="forward", nvmax=nvmax_value)

k=5
folds= sample(rep(1:k, length.out = nrow(train_set)))

n_col= ncol(dataset)-1
err= matrix(data=NA, nrow=k, ncol=n_col, dimnames=list(NULL, names(dataset)[-1]))         # 'paste' serve per convertire e inserire i numeri come stringa

curr_method= "forward"   # All possibilities: exhaustive, forward, backward, seqrep
for(j in 1:k){
  fit_cv= regsubsets(Y~., data=train_set[folds!=j, ], method=curr_method, nvmax=nvmax_value)
  
  # Loop su tutti i modelli calcolati
  for(i in 1:n_col){
    pred= predict(fit_cv, train_set[folds==j, ], id=i)    # calcolo del valore di predizione
    err[j,i]= mean( (train_set$Y[folds==j]-pred)^2 )        # calcolo dell'errore rispetto a fold e modello
  }
}

err.mean= colMeans(err)

dev.new()
par(mfrow=c(1,1))
plot(10*log10(err.mean), type="b", main="Cross validation on train set", xlab="n° indexes", ylab="MSE [dB]")
legend("topright", title="Method used", c(curr_method), col=c('black'), lty=c(1))

show_info(summary(fit.fwd))

err.fwd.index= which.min(err.mean)
pred= predict(fit.fwd, test_set, id=err.fwd.index)      # calcolo del valore di predizione
err.fwd= mean( (test_set$Y-pred)^2 )
sprintf("Min error with '%s method' is %f with %d variables", curr_method, err.fwd, err.fwd.index)

# Ridge & Lasso ----
library(glmnet)
y= as.matrix(dataset[1])
x= model.matrix(Y~., data=dataset)[, -1]  # Generazione della matrice 'X' di design

n_folds= 10

cv_out.ridge= cv.glmnet(x[train_index, ], y[train_index], alpha=0, nfolds=n_folds)
cv_out.lasso= cv.glmnet(x[train_index, ], y[train_index], alpha=1, nfolds=n_folds)

dev.new()
plot(cv_out.ridge, main="ridge")
dev.new()
plot(cv_out.lasso, main="lasso")

# Valutazione della predizione con lambda ottimo
lambda_best.ridge= cv_out.ridge$lambda.min
lambda_best.lasso= cv_out.lasso$lambda.min

pred.ridge= predict(cv_out.ridge, s='lambda.min', newx=x[test_index, ])
pred.lasso= predict(cv_out.lasso, s='lambda.min', newx=x[test_index, ])
err.ridge= mean( (pred.ridge-y[test_index])^2 )
err.lasso= mean( (pred.lasso-y[test_index])^2 )
sprintf("RIDGE : Error with lambda %f is %f", lambda_best.ridge, err.ridge)
sprintf("LASSO : Error with lambda %f is %f", lambda_best.lasso, err.lasso)

lasso_coef= predict(cv_out.lasso, s=lambda_best.lasso, type="coefficients")[1:ncol(dataset), ]
sprintf("Lasso coefficient:"); sprintf("Number null coefficient is %d", sum(lasso_coef==0)); as.matrix(lasso_coef)

# Confronto prestazioni ----
sprintf("Min error with '%s method' is %f with %d variables", "forward", err.fwd, err.fwd.index)
sprintf("RIDGE : Error with lambda %f is %f", lambda_best.ridge, err.ridge)
sprintf("LASSO : Error with lambda %f is %f", lambda_best.lasso, err.lasso)

# Coefficienti ----
as.matrix( coef(fit.fwd, id=err.fwd.index) )

# ----
cat( names(coef(fit.fwd, id=err.fwd.index)) )
cat( names(result$coefficients[which(result$coefficients!=0)]) )
