Time Series Forecasting
================
Chenyue Xu
23/04/2021

``` r
library(forecast)
library(nnfor)
library(fGarch)
library(imputeTS)
library(vars)
```

``` r
last.name = 'xu'
student.id = 1234567
```

## Scenario 1

To start this problem, we first import the data into R and convert it to
time series. Since this is a monthly data, then we will set the
frequency to 12.

``` r
# Scenario 1
# Import data
hyd <- read.table("hyd_post.txt", header = TRUE, sep = ",")[,2]
# Set frequency (it is a monthly data)
hyd.ts <- ts(hyd, frequency = 12)
plot(hyd.ts, main = "Plot of Hyd")
```

![](443-Final_files/figure-gfm/unnamed-chunk-3-1.png)<!-- -->

From the plot, we can see that seasonal component exists in the time
series, thus we will fit the time series into a SARIMA model by checking
ACF and PACF plot using eye-ball test.

``` r
# Perform eye-ball test to fit SARIMA model
acf(diff(diff(hyd.ts,lag=12)), main = "ACF plot")
```

![](443-Final_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

``` r
pacf(diff(diff(hyd.ts, lag=12)),lag.max = 60,main="PACF plot")
```

![](443-Final_files/figure-gfm/unnamed-chunk-4-2.png)<!-- -->

From the above ACF plot, we can obtain that d = 1, D = 1, q = 1 and Q =
1 by checking significant lag for both seasonal and non-seasonal
component. For PACF plot, we can see that p = 1 and P = 0 since there
exists geometric decay for seasonal component.

Therefore, we will fit hyd.ts into
**ARIMA(1,1,1)(0,1,1)**.

``` r
# Fit the time series into ARIMA(1,1,1)(0,1,1) and discuss if it is a good fit
fit2 = Arima(hyd.ts, order=c(1,1,1), seasonal = c(0,1,1))
checkresiduals(fit2)
```

![](443-Final_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

    ## 
    ##  Ljung-Box test
    ## 
    ## data:  Residuals from ARIMA(1,1,1)(0,1,1)[12]
    ## Q* = 37.796, df = 21, p-value = 0.01362
    ## 
    ## Model df: 3.   Total lags used: 24

From the ACF plot, we can see that most of the autocorrelation
coefficients lies within the two blue lines and there is no significant
trend exists in the plot which suggests that the residual of the model
follows a white noise. Meanwhile, the residulas approximately follows a
Gaussian distribution.

In addition, from the Ljung-Box test, p-value = 0.01362 \< 0.05 which
indicates that there is evidence reject that the residuals follows a
white noise.

Hence, we are not able to give any conclusion about the fitted model yet
and we still need to investigate the forecast from the fitted model.

``` r
# Forecast hyd.ts 24 steps ahead and construct 95% prediction interval
autoplot(forecast(fit2, PI=TRUE, level = 95, h=24))
```

![](443-Final_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

From the plot, we can see that the forecast from the fitted model is
able to capture the seasonality. Although we got different results about
whether the residuals from the fitted model follow a white noise by
using two testings, we are able to forecast the data using fitted model.
Thus, we can conclude that the model **ARIMA(1,1,1)(0,1,1)** is
reasonable and this forecast seems plausible.

  - Note that I also fit the time series into different models by using
    various functions such as auto.arima, nnetar and ets and forecast
    the series. Perform cross validation to compare the fitted models
    and then derive the conclusion that **ARIMA(1,1,1)(0,1,1)** has the
    least MSE.

Following is a portion of the 24 steps ahead forecast for **hyd.ts**.

``` r
head(forecast(fit2,h=24)$mean)
```

    ##         Jan      Feb      Mar      Apr      May      Jun
    ## 49 15.94133 15.95884 16.44480 17.49575 18.11793 18.30695

``` r
# Write output to table
write(forecast(fit2,h=24)$mean, 
      file = paste("Scenario1_",last.name,student.id,
                                             ".txt", sep = ""), ncolumns = 1 )
```

## Scenario 2

To start with this problem, we need to import stock1.txt to stock40.txt
by using loop function and store the data into a matrix.

``` r
# Scenario 2
# Import Data 
stock <- matrix(,nrow = 150, ncol = 40)
for (i in 1:40) {
  stock[,i] <- read.table(paste0("stock",i,".txt"),header = TRUE, sep=",")[,2]
}
```

Different approches that we are going to use are defined in the lecture:
Historical data, Risk Metrics, GARCH Normal Innovations and GARCH
Bootstrap Innovations. The following function is to determine which
approach is better to forecast the 15% quantile for stock i. We take the
index i as input and return the method with the lowest MSE which defined
in the instruction page.

``` r
# A function to decide which approach has the best performance
chooseModel <- function(i) {
  stock.ts <- ts(stock[,i])
  # Historical Data Approach
  q <- quantile(stock.ts[1:140], 0.15)*sqrt(1:10)
  error1 <- sum((stock.ts[141:150]-q)*(0.15-(stock.ts[141:150]<q)))
  
  # Risk Metrics
  riskMetrics <- 0
  lam <- .94
  sig1 <- mean(stock.ts[1:49]^2)
  for (j in 50:140) {
    signew <- lam*sig1 + (1-lam)*stock.ts[j]^2
    sig1 <- signew
  }
  q <- sqrt(signew)*qnorm(0.15)*sqrt(1:10)
  error2 <- sum((stock.ts[141:150]-q)*(0.15-(stock.ts[141:150]<q)))
  
  # GARCH(1,1)
  fit <- garchFit(~arma(0,0)+garch(1,1), stock.ts[1:140], trace = FALSE)
  sigthat <- predict(fit,1)$standardDeviation[1]
  q <- sigthat*qnorm(0.15)*sqrt(1:10)
  error3 <- sum((stock.ts[141:150]-q)*(0.15-(stock.ts[141:150]<q)))
  
  # Nonparametric-GARCH Bootstrap
  error.dis <- stock.ts[1:140]/fit@sigma.t
  q <- sigthat*quantile(error.dis, 0.15)*sqrt(1:10)
  error4 <- sum((stock.ts[141:150]-q)*(0.15-(stock.ts[141:150]<q)))
  
  m <- min(error1,error2,error3,error4)
  if (m == error1) {
    output <- 1
  } else if (m == error2) {
    output <- 2
  } else if (m == error3) {
    output <- 3
  } else {
    output <- 4
  }
  return (output)
}
```

Take STOCK1 as an example, **chooseModel(1)** returns 3 which means
GARCH Normal Innovations is the best approach to forecast 15% quantile
for stock1.

``` r
# We are going to forecast 15% quantile for stock1, set i = 1
method <- chooseModel(1)
method
```

    ## [1] 3

``` r
stock.ts <- ts(stock[,1])
fit <- garchFit(~arma(0,0)+garch(1,1), stock.ts, trace = FALSE)
summary(fit)
```

    ## 
    ## Title:
    ##  GARCH Modelling 
    ## 
    ## Call:
    ##  garchFit(formula = ~arma(0, 0) + garch(1, 1), data = stock.ts, 
    ##     trace = FALSE) 
    ## 
    ## Mean and Variance Equation:
    ##  data ~ arma(0, 0) + garch(1, 1)
    ## <environment: 0x7fc282cbdb58>
    ##  [data = stock.ts]
    ## 
    ## Conditional Distribution:
    ##  norm 
    ## 
    ## Coefficient(s):
    ##         mu       omega      alpha1       beta1  
    ## 0.00071136  0.00016202  0.58461195  0.00000001  
    ## 
    ## Std. Errors:
    ##  based on Hessian 
    ## 
    ## Error Analysis:
    ##         Estimate  Std. Error  t value Pr(>|t|)    
    ## mu     7.114e-04   1.183e-03    0.601  0.54779    
    ## omega  1.620e-04   3.864e-05    4.193 2.75e-05 ***
    ## alpha1 5.846e-01   1.933e-01    3.024  0.00249 ** 
    ## beta1  1.000e-08   7.432e-02    0.000  1.00000    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Log Likelihood:
    ##  399.1406    normalized:  2.660937 
    ## 
    ## Description:
    ##  Fri Jan 13 17:36:58 2023 by user:  
    ## 
    ## 
    ## Standardised Residuals Tests:
    ##                                 Statistic p-Value   
    ##  Jarque-Bera Test   R    Chi^2  2.411331  0.2994926 
    ##  Shapiro-Wilk Test  R    W      0.9902269 0.3853817 
    ##  Ljung-Box Test     R    Q(10)  18.74248  0.04365818
    ##  Ljung-Box Test     R    Q(15)  19.5139   0.1913833 
    ##  Ljung-Box Test     R    Q(20)  30.65802  0.05987002
    ##  Ljung-Box Test     R^2  Q(10)  6.723971  0.7512215 
    ##  Ljung-Box Test     R^2  Q(15)  9.774525  0.8336781 
    ##  Ljung-Box Test     R^2  Q(20)  13.31373  0.8635213 
    ##  LM Arch Test       R    TR^2   7.834519  0.7979248 
    ## 
    ## Information Criterion Statistics:
    ##       AIC       BIC       SIC      HQIC 
    ## -5.268541 -5.188257 -5.269915 -5.235924

``` r
# Sum of the GARCH estimated coefficients
sum(coef(fit)[3:4])
```

    ## [1] 0.584612

We have the sum of the GARCH(1,1) estimated coefficients = 0.584612
which is less than 1 and it indicates that the estimated GARCH model is
weakly stationary.

From the summary table of GARCH(1,1) model, the p-value of both
Jarque-Bera Test and Shapiro-Wilk Test \> 0.05 and this shows that there
is no strong evidence to reject the residual follows a normal
distribution.

The p-values of Ljung-Box Test for residual = 0.1913833 \> 0.05 which
indicates that there is no strong evidence reject the residual follows a
white noise with 15% quantile. The p-values of Ljung-Box Test for
residual square = 0.8336781 \> 0.05 which indicates that there is no
strong evidence reject the residual squared follows a white noise with
15% quantile.

The p-value for LM Arch Test = 0.7979248 \> 0.05, thus it indicates that
there is no strong evidence reject the residual is homoscedastic.

Thus, we can conclude that GARCH(1,1) seems to be a good fit for stock1.

``` r
# Forecast the 15% quantile of GARCH(1,1) for stock1
predict(fit, n.ahead = 10)
```

    ##    meanForecast  meanError standardDeviation
    ## 1  0.0007113589 0.01309558        0.01309558
    ## 2  0.0007113589 0.01619489        0.01619489
    ## 3  0.0007113589 0.01775798        0.01775798
    ## 4  0.0007113589 0.01861107        0.01861107
    ## 5  0.0007113589 0.01909215        0.01909215
    ## 6  0.0007113589 0.01936786        0.01936786
    ## 7  0.0007113589 0.01952724        0.01952724
    ## 8  0.0007113589 0.01961982        0.01961982
    ## 9  0.0007113589 0.01967374        0.01967374
    ## 10 0.0007113589 0.01970519        0.01970519

``` r
# Plot the stock1 time series
plot(stock.ts, main = "Forecast of GARCH(1,1)")
# Add quantile line to the plot
lines(fit@sigma.t*qnorm(0.15), col="red")
# Add forecast quantile line to the plot
lines(150+(1:10), predict(fit,1)$standardDeviation[1]*qnorm(0.15)*sqrt(1:10), col="blue")
legend(10, 0.08, legend=c("stock1", "15% quantile", "forecast 15% quantile"),
       col=c("black", "red", "blue"), lty=1, cex=0.8)
```

![](443-Final_files/figure-gfm/unnamed-chunk-13-1.png)<!-- -->

From the forecast plot, we can see that we are able to capture the trend
of 15% quantile of the time series and thus we can conclude that the
forecast seems plausible.

Following are the forecast for 15% quantile for
    stock1.

``` r
predict(fit,1)$standardDeviation[1]*qnorm(0.15)*sqrt(1:10)
```

    ##  [1] -0.01357269 -0.01919469 -0.02350859 -0.02714539 -0.03034946 -0.03324617
    ##  [7] -0.03590997 -0.03838937 -0.04071808 -0.04292062

## Scenario 3 and Scenario 4

To start with this problem, we first import the data and related series
that may be helpful for the following imputation and prediction.

``` r
# Scenario 3 and 4
# Import data
prodTarget <- read.table("prod_target.txt",header = TRUE, sep = ",")[,3]
prodTarget.ts <- ts(prodTarget, start=c(1956,1), frequency = 12)

# Import temp data and set this time series the same start date with prodTarget
temp <- read.table("temp.txt",header = TRUE, sep = ",")[,4]
temp.ts <- ts(temp,start=c(1943,11), frequency = 12)
temp.ts <- window(temp.ts,c(1956,1))

# Define the missing value range
mis.t <- 201:230
```

Consider that the monthly mean high temperatures may affect the amount
of beer produced in Australia, thus we will consider temp as a regressor
when we are fitting the model.

``` r
# Fitted model
m2 <- auto.arima(prodTarget.ts,xreg=temp.ts)
checkresiduals(m2)
```

![](443-Final_files/figure-gfm/unnamed-chunk-16-1.png)<!-- -->

    ## 
    ##  Ljung-Box test
    ## 
    ## data:  Residuals from Regression with ARIMA(4,1,3)(1,1,2)[12] errors
    ## Q* = 28.923, df = 13, p-value = 0.006712
    ## 
    ## Model df: 11.   Total lags used: 24

``` r
resid <- na_remove(m2$residuals)
```

We use auto.arima to fit time series **prodTarget.ts** and we obtain a
model **ARIMA(4,1,3)(1,1,2)**. From the ACF plot, we can see that most
of the autocorrelation coefficients lies within the two blue lines and
there is no significant trend exists in the plot which suggests that the
residual of the model follows a white noise. Meanwhile, the residulas
approximately follows a Gaussian distribution.

In addition, from the Ljung-Box test, p-value = 0.006712 \< 0.05 which
indicates that there is some evidence reject that the residuals follows
a white noise.

Hence, we are not able to give any conclusion about the fitted model yet
and we still need to investigate the forecast from the fitted model.

We then compute the missing value using Kalman smoothing and plot a 95%
prediction interval for the imputations.

From the plot, we can see that the imputations and 95% prediction
interval are able to capture the seasonality of time series, thus we can
conclude that this 95% prediction interval seems plausible and the
imputations using Kalman filter seems valid.

``` r
# Impute missing value using Kalman filter
imp.prod.kal2 = na_kalman(prodTarget.ts, model = m2$model)
ggplot_na_imputations(prodTarget.ts, imp.prod.kal2)
```

![](443-Final_files/figure-gfm/unnamed-chunk-17-1.png)<!-- -->

``` r
# Plot 95% prediction interval for imputation
plot(as.numeric(imp.prod.kal2), type='l', main="Imputed Series", ylab = "")
lines(mis.t,imp.prod.kal2[mis.t]+ 2*sd(resid),col=4,lty=2)
lines(mis.t,imp.prod.kal2[mis.t]- 2*sd(resid),col=4,lty=2)
```

![](443-Final_files/figure-gfm/unnamed-chunk-17-2.png)<!-- -->

Following is a portion of the imputation for the missing value in
prodTarget.

``` r
# Generate ouput
imput3 <- imp.prod.kal2[mis.t]
head(imput3)
```

    ## [1] 134.2250 155.6683 167.6643 175.0436 145.5520 145.9323

``` r
# Write output to table
write(imput3, file = paste("Scenario3_",last.name,student.id,
                           ".txt", sep = ""), ncolumns = 1 )
```

Since our fitted model of prodTarget.ts uses temp as a regressor, thus
when we are doing forecast, we need to first predict temp.ts 24 steps
ahead and then do the forecast for prodTarget.ts.

``` r
# Forecast prodTarget.ts 24 steps ahead
temp.model <- auto.arima(temp.ts)
temp.pred <- forecast(temp.model,24)$mean
autoplot(forecast(m2,24,xreg=temp.pred,level=95))
```

![](443-Final_files/figure-gfm/unnamed-chunk-19-1.png)<!-- -->

From the forecast plot, we can see that we are able to capture the
seasonality of the series and thus we can conclude that this fitted
model seems reasonable and the forecast seems plausible.

Following is a portion of the forecast for 24 steps ahead for series
prodTarget.ts.

``` r
# Generate output
forecast4 <- forecast(m2,24,xreg = temp.pred)$mean
head(forecast4)
```

    ##           Apr      May      Jun      Jul      Aug      Sep
    ## 1992 154.0687 144.7021 123.8740 147.2274 146.3937 138.4680

``` r
# Write output for prediction part
write(forecast4, file = paste("Scenario4_",last.name,student.id,
                              ".txt", sep = ""), ncolumns = 1 )
```

## Scenario 5

To start with this problem, we first import the three datasets and set
the frequency to 48 since it is a half hourly data.

``` r
# Scenario 5
# Import data and convert into time series
c1 <- ts(read.table("pollutionCity1.txt", header = TRUE, sep = ",")[,2],
         frequency = 48)
c2 <- ts(read.table("pollutionCity2.txt", header = TRUE, sep = ",")[,2], 
         frequency = 48)
c3 <- ts(read.table("pollutionCity3.txt", header = TRUE, sep = ",")[,2], 
         frequency = 48)
```

Since the three time series may have correlation with each other, then
we combine the series to create a multivariate time series.

``` r
# Combine three series
dat.mat.full=cbind(as.numeric(c1),as.numeric(c2),as.numeric(c3))
colnames(dat.mat.full) = c("City1", "City2", "City3")
```

We need to use Vector Auto Regression (VAR) to forecast as we are
dealing with multivariate time series.

``` r
# Computes AICs for VAR models up to lag 20
x=VARselect(dat.mat.full, lag.max = 20)$criteria[1,] 
plot(x, main="AIC as function of maximal lag", xlab= "LAG", ylab="AIC",cex=2)
```

![](443-Final_files/figure-gfm/unnamed-chunk-23-1.png)<!-- -->

``` r
which.min(x)
```

    ## 17 
    ## 17

By comparing the AIC, we select the VAR(p) model where at p, AIC is the
smallest and this indicates that this model may have the best
performance other than other VAR model.

``` r
var17 <- VAR(dat.mat.full, p = 17, type = "const")
# Prediction
var17_prd <- predict(var17, n.ahead = 336, ci = 0.95)
par(mar=c(1,1,1,1))
# Forecast plot
plot(var17_prd)
```

![](443-Final_files/figure-gfm/unnamed-chunk-24-1.png)<!-- -->

From the forecast plot, we can see that although we have already chose
the model with least AIC which should have the best performance among
all the potential models. However, neither of the forecast for any of
the variates has indicated the seasonal component of the time series in
the prediction. Thus, we can conclude that this forecast does not seem
very plausible.

One reason that may cause this problem is that time series City1, City2
and City3 have seasonal component and in order to forecast, we need to
eliminate the seasonal component for the series or try to predict the
series with seasonal component. By doing this, it may improve the
performance of forecast.

We then perform a serial test on Var(17) model to test if serial
correlation exists.

``` r
# Multivariate Portmanteau Test
serial.test(var17,lags.pt=20)
```

    ## 
    ##  Portmanteau Test (asymptotic)
    ## 
    ## data:  Residuals of VAR object var17
    ## Chi-squared = 47.811, df = 27, p-value = 0.008059

From the output, we have p-value = 0.008059 \< 0.05 which indicates that
there is evidence reject that the model does not exist serial
correlation.

According to the conclusion that we just derived, one possible approach
to make the forecast better is to eliminate the serial correlation
between the variates.

Following is a portion of the forecast 336 steps ahead for each series.

``` r
# Create an empty output matrix and load the data into the table
output <- matrix(,nrow = 336,ncol=3)
output[,1] <- var17_prd$fcst$City1[,1]
output[,2] <- var17_prd$fcst$City2[,1]
output[,3] <- var17_prd$fcst$City3[,1]
# Show the result for forecast 336 steps ahead for each series
head(output)
```

    ##          [,1]     [,2]     [,3]
    ## [1,] 32.26801 39.49478 51.92345
    ## [2,] 31.42221 41.43402 49.76210
    ## [3,] 31.61211 43.57193 49.04253
    ## [4,] 33.42171 47.78575 47.85753
    ## [5,] 33.94215 48.63660 49.24666
    ## [6,] 36.62126 47.52824 52.60613

``` r
# Write output
write.table(output, file = paste("Scenario5_",last.name,student.id,
                                    ".txt", sep = ""), sep ="," , col.names = F, row.names = F )
```

## Appendix

  - Most of the code have included in the report part except the code of
    generating output for Scenario 2.

<!-- end list -->

``` r
# Constrcut an empty output matrix
output <- matrix(,nrow = 10, ncol = 40)
# Loop through 1 to 40 and choose the best approach for each stock
for (i in 1:40) {
  method <- chooseModel(i)
  if (method == 1) {
    output[,i] <- quantile(stock[,i],0.15)*sqrt(1:10)
  } else if (method == 2) {
    riskMetrics <- 0
    lam <- .94
    sig1 <- mean(stock[1:49]^2)
    for (j in 50:150) {
      signew <- lam*sig1 + (1-lam)*stock[j,i]^2
      sig1 <- signew
    }
    output[,i] <- sqrt(signew)*qnorm(0.15)*sqrt(1:10)
  } else if (method == 3) {
    m1 <- garchFit(~arma(0,0)+garch(1,1), stock[,i], trace = FALSE)
    sigthat <- predict(m1,1)$standardDeviation[1]
    output[,i] <- sigthat*qnorm(0.15)*sqrt(1:10)
  } else {
    error.dis <- stock[,i]/m1@sigma.t
    output[,i] <- sigthat*quantile(error.dis, 0.15)*sqrt(1:10)
  }
}
write.table(output, file = paste("Scenario2_",last.name,student.id,
                                 ".txt", sep = ""), sep ="," , col.names = F, row.names = F )
```
