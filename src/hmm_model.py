from hmmlearn.hmm import GaussianHMM
import numpy as np
import pandas as pd

class RegimeHMM:
    def __init__(self, n_states=3, covariance_type='full', n_iter=100):
        self.n_states = n_states
        self.model = GaussianHMM(n_components=n_states, 
                                 covariance_type=covariance_type, 
                                 n_iter=n_iter, 
                                 random_state=42)
        self.labels = None

    def fit(self, X):
        """
        Fit the HMM model. 
        Important: X should be scaled before calling this, 
        but I'll add a check just in case or just assume the user knows.
        Actually, let's just use it as is for now, but I'll add a note.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        self.model.fit(X)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def get_regime_labels(self, X, feature_names):
        """
        Sort states by SPY return to assign 'Bull', 'Bear', 'Crisis'
        """
        # This is a bit tricky, need to find the SPY return column
        spy_col = [i for i, name in enumerate(feature_names) if 'SPY' in name and 'ret' in name]
        if not spy_col:
            return {i: f"State {i}" for i in range(self.n_states)}
        
        spy_idx = spy_col[0]
        means = self.model.means_[:, spy_idx]
        sorted_indices = np.argsort(means)
        
        labels = {
            sorted_indices[0]: 'Crisis',
            sorted_indices[1]: 'Bear',
            sorted_indices[2]: 'Bull'
        }
        return labels

if __name__ == "__main__":
    # Dummy test
    X = np.random.randn(100, 5)
    hmm = RegimeHMM()
    hmm.fit(X)
    print("Model fitted.")
    print(f"Transition matrix:\n{hmm.model.transmat_}")
