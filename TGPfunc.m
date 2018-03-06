function loglik = TGPfunc(hyp,model)
    hyp = hyp'.*(model.bounds.ub-model.bounds.lb)+model.bounds.lb;
    model.hyp = exp(hyp);
    model = model.eval();
    loglik = -model.loglik;
end