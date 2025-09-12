from jax import jit
import jax.numpy as jnp

@jit
def mse(y,ytrue):
    return ((y - ytrue) ** 2).mean()
@jit    
def compute_bin_acc(y,ytrue):
   return 100*(jnp.mean(jnp.sign(y)==ytrue))

def get_model_performance(perf_log ,model, params, Xtr, Xtst, ytr, ytst, task):
    yhat = model.apply(params, Xtr)
    yPre = model.apply(params, Xtst)
    train_error = mse(yhat, ytr)
    test_error = mse(yPre, ytst)
    perf_log['train_loss'].append(train_error)
    perf_log['test_loss'].append(test_error)
    
    if task == 'classification':
       train_acc = compute_bin_acc(yhat, ytr)
       test_acc = compute_bin_acc(yPre, ytst)
       perf_log['train_acc'].append(train_acc)
       perf_log['test_acc'].append(test_acc)
    return perf_log
