# configs/ethiopia_downstream_config.py

config = {
    # Paths
    "representations_npz": "/maps/zf281/btfm4rs/data/representation/ethiopia/representations.npz",
    "output_dir": "/maps/zf281/btfm4rs/data/downstream/ethiopia/logs",
    
    # Dataset parameters
    "samples_per_class": 10,  # Number of samples per class for training (few-shot setting)
    
    # Classifier type
    # Options: 'mlp' (PyTorch linear layer), 'logistic' (LogisticRegression), 'knn1' (K=1), 'knn3' (K=3)
    "classifier": "knn1",  
    
    # Training parameters
    "batch_size": 64,
    "epochs": 100,
    "lr": 0.001,
    "weight_decay": 0.01,
    "num_workers": 4,
    
    # Experiment settings
    "num_experiments": 500,  # Number of experiments to run with different random seeds
    
    # Class names
    "class_names": ["Teff", "maize", "barley", "wheat"]
}