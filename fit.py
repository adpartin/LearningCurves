import rpy2.robjects as robjects

# def biased_powerlaw(x, a, b, c):
#     return a + b*(x**(c))

# def params(x: robjects.IntVector, y: robjects.FloatVector) -> tuple:
#     # script = '\'fit.R\''
#     # script = '\'fit.R\''
#     script = '\'../fit.R\''
#     robjects.r('''source({})'''.format(script))
#     get_params = robjects.globalenv['model_param']
#     a, b, c = get_params(x, y)
#     return (a, b, c-0.5)


def biased_powerlaw(x, alpha, beta, gamma):
    return alpha * x**(beta) + gamma


# def params(x: robjects.IntVector, y: robjects.FloatVector) -> tuple:
def fit_params(x, y) -> tuple:  
    x = robjects.IntVector(list(x))
    y = robjects.FloatVector(list(y))
    # script = '\'fit.R\''
    # script = '\'fit.R\''
    script = '\'../fit.R\''
    robjects.r('''source({})'''.format(script))
    get_params = robjects.globalenv['model_param']
    a, b, c = get_params(x, y)
    
    # gamma, alpha, beta = get_params(x, y)
    prms_dct = {}
    prms_dct = {}
    prms_dct['alpha'], prms_dct['beta'], prms_dct['gamma'] = b, c-0.5, a
    # return (a, b, c-0.5)
    return prms_dct
    