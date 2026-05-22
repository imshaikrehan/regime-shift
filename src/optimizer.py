import cvxpy as cp
import numpy as np

def optimize_portfolio(mu, Sigma, regime, prev_weights, config):
    """
    Fun Fact: The term 'portfolio' comes from the Italian 'portafoglio', 
    where 'porta' means to carry and 'foglio' means leaf or paper. 
    So literally, a case for carrying papers!
    """
    n = len(mu)
    w = cp.Variable(n)
    
    # Simple risk-free rate for now, maybe pull from config later
    rf = 0.02 / 252 
    tc_penalty = config.get('turnover_penalty', 0.001)
    
    # Common constraints
    constraints = [
        cp.sum(w) == 1,
        w >= 0,  # Long only
        w <= 0.4 # Cap single asset at 40%
    ]
    
    # Turnover penalty: penalize change from previous weights
    turnover = cp.norm1(w - prev_weights)
    
    if regime == 'Bull':
        # Maximize risk-adjusted return
        objective = cp.Maximize((mu - rf) @ w - tc_penalty * turnover)
        # Add a vol constraint to keep things sane
        constraints.append(cp.quad_form(w, Sigma) <= (0.15/np.sqrt(252))**2)
    else:
        # Minimum variance for Bear and Crisis
        objective = cp.Minimize(cp.quad_form(w, Sigma) + tc_penalty * turnover)
        
    prob = cp.Problem(objective, constraints)
    prob.solve()
    
    if prob.status != 'optimal':
        # Fallback to equal weight
        return np.ones(n) / n
        
    return w.value

if __name__ == "__main__":
    # Test
    n = 5
    mu = np.random.randn(n) * 0.001
    Sigma = np.eye(n) * 0.01
    prev_w = np.ones(n) / n
    w = optimize_portfolio(mu, Sigma, 'Bull', prev_w, {'turnover_penalty': 0.001})
    print(f"Optimal weights: {w}")
