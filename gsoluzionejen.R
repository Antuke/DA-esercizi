# Caricamento dataset ----

rm(list=ls())
graphics.off()
library(car)
dataset= read.csv("./RegressionDA240110.csv")
names(dataset)

sum( is.na(dataset$Y) )
dataset= na.omit(dataset)

# Divisione train e test ----
train_index= sample(1:nrow(dataset), size=0.8*nrow(dataset)) # size: numero di elementi da scegliere
test_index= -train_index

# Valutazione preliminare collinearità e VIF ----
library(corrplot)
correlation= round(cor(dataset), digits=2)
dev.new()
corrplot.mixed(correlation, order='original', number.cex=1, upper="ellipse")

fit= lm(Y~., data=dataset)
vif_values= vif(fit)
as.matrix(vif_values)

# Funzioni di utilità ----

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

# Stepwise forward ----
library(leaps)
nvmax_value= dim(dataset)[2]-1

fit.fwd= regsubsets(Y~., data=dataset[train_index, ], method="forward", nvmax=nvmax_value)
show_info(summary(fit.fwd))

k=10
folds= sample(x=1:k, size=nrow(dataset[train_index, ]), replace=TRUE)           # campiona casualmente 'x' per 'size' volte

n_col= ncol(dataset)-1
err= matrix(data=NA, nrow=k, ncol=n_col, dimnames=list(NULL, names(dataset)[-1]))         # 'paste' serve per convertire e inserire i numeri come stringa

curr_method= "forward"    # All possibilities: exhaustive, forward, backward, seqrep
for(j in 1:k){
  fit_cv= regsubsets(Y~., data=dataset[folds!=j, ], method=curr_method, nvmax=nvmax_value)
  
  # Loop su tutti i modelli calcolati
  for(i in 1:n_col){
    pred= predict(fit_cv, dataset[folds==j, ], id=i)      # calcolo del valore di predizione
    err[j,i]= mean( (dataset$Y[folds==j]-pred)^2 )        # calcolo dell'errore rispetto a fold e modello
  }
}

err.mean= colMeans(err)
err.fwd= err.mean

dev.new()
par(mfrow=c(1,1))
plot(10*log10(err.mean), type="b", main="Cross validation on train set", xlab="n° indexes", ylab="MSE [dB]")
legend("topright", title="Method used", c(curr_method), col=c('black'), lty=c(1))

index_min_err= which.min(err.mean)
sprintf("Min error with '%s method' is %f with %d variables", curr_method, err.mean[index_min_err], index_min_err)

# Stepwise backward ----
library(leaps)
nvmax_value= dim(dataset)[2]-1

fit.bwd= regsubsets(Y~., data=dataset[train_index, ], method="backward", nvmax=nvmax_value)
show_info(summary(fit.fwd))

k=10
folds= sample(x=1:k, size=nrow(dataset[train_index, ]), replace=TRUE)           # campiona casualmente 'x' per 'size' volte

n_col= ncol(dataset)-1
err= matrix(data=NA, nrow=k, ncol=n_col, dimnames=list(NULL, names(dataset)[-1]))         # 'paste' serve per convertire e inserire i numeri come stringa

curr_method= "backward"   # All possibilities: exhaustive, forward, backward, seqrep
for(j in 1:k){
  fit_cv= regsubsets(Y~., data=dataset[folds!=j, ], method=curr_method, nvmax=nvmax_value)
  
  # Loop su tutti i modelli calcolati
  for(i in 1:n_col){
    pred= predict(fit_cv, dataset[folds==j, ], id=i)      # calcolo del valore di predizione
    err[j,i]= mean( (dataset$Y[folds==j]-pred)^2 )        # calcolo dell'errore rispetto a fold e modello
  }
}

err.mean= colMeans(err)
err.bwd= err.mean

dev.new()
par(mfrow=c(1,1))
plot(10*log10(err.mean), type="b", main="Cross validation on train set", xlab="n° indexes", ylab="MSE [dB]")
legend("topright", title="Method used", c(curr_method), col=c('black'), lty=c(1))

index_min_err= which.min(err.mean)
sprintf("Min error with '%s method' is %f with %d variables", curr_method, err.mean[index_min_err], index_min_err)




# ridge & lasso ----
# (alpha=0 the ridge penalty, and alpha=1 is the lasso penalty)
library(glmnet)
y= as.matrix(dataset[1])
x= model.matrix(Y~., data=dataset)[, -1]  # Generazione della matrice 'X' di design

lambda_vec= 10^seq(10, -2, length=100)    # va da 10^(10) a 10^(-2)
n_folds= 10

ridge= glmnet(x[train_index, ], y[train_index, ], alpha=0, lambda=lambda_vec, nfolds=n_folds)
lasso= glmnet(x[train_index, ], y[train_index, ], alpha=1, lambda=lambda_vec, nfolds=n_folds)

cv_out.ridge= cv.glmnet(x[train_index, ], y[train_index], alpha=0)
cv_out.lasso= cv.glmnet(x[train_index, ], y[train_index], alpha=1)

dev.new()
plot(cv_out.ridge, main="ridge")
dev.new()
plot(cv_out.lasso, main="lasso")

# Valutazione della predizione con lambda ottimo
lambda_best.ridge= cv_out.ridge$lambda.min
lambda_best.lasso= cv_out.lasso$lambda.min

pred.ridge= predict(ridge, s=lambda_best.ridge, newx=x[test_index, ])
pred.lasso= predict(lasso, s=lambda_best.lasso, newx=x[test_index, ])
err.ridge= mean( (pred.ridge-y[test_index])^2 )
err.lasso= mean( (pred.lasso-y[test_index])^2 )
sprintf("RIDGE : Error with lambda %f is %f", lambda_best.ridge, err.ridge)
sprintf("LASSO : Error with lambda %f is %f", lambda_best.lasso, err.lasso)

lasso_coef= predict(cv_out.lasso, s=lambda_best.lasso, type="coefficients")[1:ncol(dataset), ]
sprintf("Lasso coefficient:"); sprintf("Number null coefficient is %d", sum(lasso_coef==0)); as.matrix(lasso_coef)

# Scelta modello ----


index_min_err= which.min(err.fwd)
sprintf("Min error with '%s method' is %f with %d variables", "farward", err.fwd[index_min_err], index_min_err)
sprintf("Perfomance on test-set is: %f",  err.fwd[index_min_err] )

index_min_err= which.min(err.bwd)
sprintf("Min error with '%s method' is %f with %d variables", "backward", err.bwd[index_min_err], index_min_err)
sprintf("Perfomance on test-set is: %f", err.bwd[index_min_err])


sprintf("RIDGE : Error with lambda %f is %f", lambda_best.ridge, err.ridge)
sprintf("LASSO : Error with lambda %f is %f", lambda_best.lasso, err.lasso)



# Scrittura parametri ----
as.matrix( coef(fit.fwd, id=14) )

