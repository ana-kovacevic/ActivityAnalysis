

data = [1 2 1 1 2 2 2 1 2 3 3 2 3 2 1 2 2 3 4 5 5 3 3 2 6 6 5 6 4 3 4 4 4 4 4 4 3 3 2 2];
model = hmmFit(data, 2, 'discrete');
model.pi = 0.6661    0.3339;
model.A =
    0.8849    0.1151
    0.1201    0.8799
model.emission.T =
    0.2355    0.5232    0.2259    0.0052    0.0049    0.0053
    0.0053    0.0449    0.2204    0.4135    0.1582    0.1578
logLike = hmmLogprob(model,data);
logLike =  -55.8382