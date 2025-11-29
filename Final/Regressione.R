# Caricamento dataset ----
dataset= read.csv("./RegressionDSDA250130.csv")
names(dataset)

# Verifica presenza valori NULL in Y e cancellazione (se necessario)
sum( is.na(dataset$Y) )

# Verifica correlazione & collinearità ----
library(car)
library(corrplot)

# Verifvifica correlazione
correlation= round(cor(dataset), digits=2)
dev.new()
corrplot.mixed(correlation, order='original', number.cex=1, upper="ellipse")

# COMMENTO:
# Risulta che le variabili sono tra loro poco correlate
# Si può notare che la variabile Y è molto correlata con le variabili X2, X4, X18.
#   Per questo motivo dovrebbero essere utili nella predizione

# Verifica collinearità
fit= lm(Y~., data=dataset)
vif_values= vif(fit)
as.matrix(vif_values)

# COMMENTO:
# Nessun valore supera 5 quindi non presenta collinearità
# In verità nessun valore supera 2

# Divisione in train e test ----
size_train= 0.7
train_index= sample(1:nrow(dataset), size=size_train*nrow(dataset))
test_index= -train_index

train_data= dataset[train_index, ]
test_data=  dataset[test_index, ]

# Best subset selection (con BIC) ----
library(leaps)
nvmax_value= dim(dataset)[2]-1

fit.full= regsubsets(Y~., data=train_data, method="exhaustive", nvmax=nvmax_value)
fit.full.summary= summary(fit.full)
fit.full.min_index= which.min(fit.full.summary$bic)

dev.new()
plot(fit.full.summary$bic, type='l', lwd='1', xlab="n° variables", ylab="BIC")
points(which.min(fit.full.summary$bic), min(fit.full.summary$bic), type='p', pch=16, col="red")  
points(3, fit.full.summary$bic[3], type='p', pch=16 , col="blue")
legend(
  'topleft',
  c('BIC', 'min value', 'value at 3'),
  col=c('black', 'red', 'blue'),
  lty=c(1, NA, NA),
  pch=c(NA, 16, 16)
)

# COMMENTO:
# Dato che i regressori più significativi dall'analisi della collinearità sono risultati
# essere tre, si può notare come l'errore prendendo solamente tre regressori (che
# con molta probabilità saranno X2, X4, X18) è simile a quella minima risultante
# dal BIC

sprintf("Best subset selection with index %d", fit.full.min_index)
sprintf("Coefficient:")
as.matrix( coef(fit.full, id=fit.full.min_index) )

# Stepwise backward (cross-validation 5 folds) ----

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

k=5
folds= sample(rep(1:k, length.out=nrow(train_data)))

n_col= ncol(dataset)-1
err= matrix(data=NA, nrow=k, ncol=n_col)

for(j in 1:k){
  fit_cv= regsubsets(Y~., data=train_data[folds!=j, ], method="backward", nvmax=nvmax_value)
  
  # Loop su tutti i modelli calcolati
  for(i in 1:n_col){
    pred= predict(fit_cv, train_data[folds==j, ], id=i)    # calcolo del valore di predizione
    err[j,i]= mean( (train_data$Y[folds==j]-pred)^2 )        # calcolo dell'errore rispetto a fold e modello
  }
}

err.mean= colMeans(err)

dev.new()
par(mfrow=c(1,1))
plot(10*log10(err.mean), type="b", main="Cross validation on train set", xlab="n° indexes", ylab="MSE [dB]")
legend("bottomright", title="Method used", c("backward"), col=c('black'), lty=c(1))

# COMMENTO:
# Verificando con divisioni differenti del dataset risulta che nella maggior parte
# dei casi che l'errore minimo risulta essere presente spesso a 3 (i regressori sono
# come notato prima X2, X4, X18)

fit.bwd.min_index= which.min(err.mean)
fit.bwd= regsubsets(Y~., data=train_data, method="backward", nvmax=fit.bwd.min_index)
sprintf("Best backward selection with index %d", fit.bwd.min_index)
sprintf("Coefficient:")
as.matrix( coef(fit.bwd, id=fit.bwd.min_index) )

# Ridge ----
library(glmnet)
y= as.matrix(dataset[1])
x= model.matrix(Y~., data=dataset)[, -1]  # Generazione della matrice 'X' di design

n_folds= 10

cv.ridge= cv.glmnet(x[train_index, ], y[train_index], alpha=0, nfolds=n_folds)

dev.new()
plot(cv.ridge)

sprintf("Best ridge with lambda %f (or log(lambda) %f)", cv.ridge$lambda.min, log(cv.ridge$lambda.min))
sprintf("Coefficient:")
as.matrix( predict(cv.ridge, s=cv.ridge$lambda.min, type="coefficients")[1:ncol(dataset), ] )

# Lasso ----
cv.lasso= cv.glmnet(x[train_index, ], y[train_index], alpha=1, nfolds=n_folds)

dev.new()
plot(cv.lasso)

sprintf("Best lasso with lambda %f (or log(lambda) %f)", cv.lasso$lambda.min, log(cv.lasso$lambda.min))
sprintf("Coefficient:")
as.matrix( predict(cv.lasso, s=cv.lasso$lambda.min, type="coefficients")[1:ncol(dataset), ] )

# MSE sul test set ----
pred= predict(fit.full, newdata=test_data, id=fit.full.min_index)
err.full= mean( (pred-y[test_index])^2 )

pred= predict(fit.bwd, newdata=test_data, id=fit.bwd.min_index)
err.bwd= mean( (pred-y[test_index])^2 )

pred= predict(cv.ridge, s='lambda.min', newx=x[test_index, ])
err.ridge= mean( (pred-y[test_index])^2 )

pred= predict(cv.lasso, s='lambda.min', newx=x[test_index, ])
err.lasso= mean( (pred-y[test_index])^2 )

sprintf("Best subset with index %d has error : %f", fit.full.min_index, err.full)
sprintf("Stepwise backward with index %d has error : %f", fit.bwd.min_index, err.bwd)
sprintf("Ridge with lambda %f has error : %f", cv.ridge$lambda.min, err.ridge)
sprintf("Lasso with lambda %f has error : %f", cv.lasso$lambda.min, err.lasso)

