if(!require(broom)){
  install.packages("broom")
  library(broom)
}

if(!require(dplyr)){
  install.packages("dplyr")
  library(dplyr)
}

# self-defined self-starting fct
# for more details plz refer to 
# Predicting accuracy on large datasets from smaller pilot data
weightfn_selfstart = function (d) { d$training.examples }
plaw_selfStart = selfStart(~ a + b*(x**(c-0.5)),
                           function(mCall, data, LHS) {
                             xy = sortedXyData(mCall[["x"]], LHS, data);
                             d = xy %>% mutate(training.examples=x, error=y);
                             lmFit = lm(error ~ I(1/sqrt(training.examples)), data=d, weights=weightfn_selfstart(d));
                             coefs = coef(lmFit);
                             a = coefs[1]; # intercept
                             b = coefs[2]; # factor
                             value = c(a, b, 0);
                             names(value) = mCall[c("a", "b", "c")];
                             return(value);
                           },
                           c("a","b","c"));

bionomial = function (d) { d$training.examples/(d$error*(1-d$error))}

get_model = function (data) {
  # eplaw models: error = a + b*training.examples^c
  eplaw_models = data %>%
    do(model = nls(error ~ plaw_selfStart(training.examples,a,b,c),
                   data=.,
                   control=nls.control(warnOnly=TRUE,maxiter=100000,tol=1e-4,minFactor=1e-7),
                   weights=bionomial(.)));
  return(eplaw_models$model[[1]])
}

model_param <- function (x, y) {
  data_train = do.call(rbind, Map(data.frame, 
                                  training.examples=x, 
                                  error=y))
  model = get_model(data_train)
  return (coef(model))
}
