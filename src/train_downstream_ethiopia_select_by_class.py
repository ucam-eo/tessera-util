import os
import time
import argparse
import logging
import numpy as np
import pandas as pd
import torch
import tqdm
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

def parse_args():
    parser = argparse.ArgumentParser(description="Ethiopia Downstream Classification with Pre-computed Representations")
    parser.add_argument('--config', type=str, default="configs/ethiopia_downstream_via_rep_config.py", help="Path to config file")
    parser.add_argument('--no_wandb', default=True, action='store_false', help="Disable wandb logging")
    return parser.parse_args()

def load_config_module(config_file_path):
    """Load a config module from a Python file."""
    import importlib.util
    import sys
    
    spec = importlib.util.spec_from_file_location("my_dynamic_config", config_file_path)
    config_module = importlib.util.module_from_spec(spec)
    sys.modules["my_dynamic_config"] = config_module
    spec.loader.exec_module(config_module)
    return config_module

class LinearProbeHead(nn.Module):
    """Simple linear classification head for linear probing."""
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.fc(x)

class EthiopiaRepresentationDataset(Dataset):
    """Dataset for Ethiopia crop classification using pre-computed representations."""
    def __init__(self, representations, labels, class_names, samples_per_class, split='train', random_state=42):
        self.representations = representations
        self.labels = labels
        self.class_names = class_names
        self.num_classes = len(class_names)
        
        # Create class indices
        class_indices = [np.where(labels == i)[0] for i in range(self.num_classes)]
        
        if split == 'train':
            # Select samples_per_class samples for each class for training
            self.indices = []
            for i in range(self.num_classes):
                if len(class_indices[i]) <= samples_per_class:
                    self.indices.extend(class_indices[i])
                else:
                    # Randomly select samples_per_class samples
                    np.random.seed(random_state)
                    selected_indices = np.random.choice(class_indices[i], samples_per_class, replace=False)
                    self.indices.extend(selected_indices)
        else:  # 'val'
            # Use all remaining samples for validation
            self.indices = []
            for i in range(self.num_classes):
                if len(class_indices[i]) > samples_per_class:
                    # Randomly select samples for training (to exclude from validation)
                    np.random.seed(random_state)
                    train_indices = set(np.random.choice(class_indices[i], samples_per_class, replace=False))
                    # Use the rest for validation
                    val_indices = [idx for idx in class_indices[i] if idx not in train_indices]
                    self.indices.extend(val_indices)
        
        print(f"{split} dataset size: {len(self.indices)}")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        index = self.indices[idx]
        return self.representations[index], self.labels[index]
    
    def get_numpy_data(self):
        """Get the entire dataset as numpy arrays for sklearn models"""
        X = np.array([self.representations[i] for i in self.indices])
        y = np.array([self.labels[i] for i in self.indices])
        return X, y

def get_classifier(classifier_type, input_dim, num_classes, random_state=42):
    """Create a classifier based on the specified type"""
    if classifier_type == 'mlp':
        return LinearProbeHead(input_dim=input_dim, num_classes=num_classes)
    elif classifier_type == 'logistic':
        return LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            C=1.0,
            solver='lbfgs',
            multi_class='multinomial'
        )
    elif classifier_type == 'knn1':
        return KNeighborsClassifier(n_neighbors=1, weights='uniform')
    elif classifier_type == 'knn3':
        return KNeighborsClassifier(n_neighbors=3, weights='distance')
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")

def is_sklearn_model(classifier_type):
    """Check if the classifier is an sklearn model"""
    return classifier_type in ['logistic', 'knn1', 'knn3']

def get_detailed_metrics(y_true, y_pred, class_names):
    """Get detailed metrics including per-class precision, recall, and F1"""
    metrics = {}
    
    # Overall metrics
    metrics['overall_accuracy'] = accuracy_score(y_true, y_pred)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)  # Added balanced accuracy
    metrics['overall_precision'] = precision_score(y_true, y_pred, average='weighted')
    metrics['overall_recall'] = recall_score(y_true, y_pred, average='weighted')
    metrics['overall_f1'] = f1_score(y_true, y_pred, average='weighted')
    
    # Per-class metrics
    class_report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    for i, class_name in enumerate(class_names):
        metrics[f'class_{i+1}_precision'] = class_report[class_name]['precision']
        metrics[f'class_{i+1}_recall'] = class_report[class_name]['recall']
        metrics[f'class_{i+1}_f1'] = class_report[class_name]['f1-score']
    
    return metrics

def main():
    args = parse_args()
    
    # Load configuration
    config_module = load_config_module(args.config)
    config = config_module.config
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set up wandb
    use_wandb = False
    if not args.no_wandb:
        try:
            import wandb
            run_name = f"ethiopia_crop_repr_{time.strftime('%Y%m%d_%H%M%S')}"
            wandb_run = wandb.init(project="btfm-downstream", name=run_name, config=config)
            use_wandb = True
        except ImportError:
            logging.warning("wandb not installed, will not use wandb logging")
    
    # Create output directories
    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs(os.path.join(config['output_dir'], "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(config['output_dir'], "logs"), exist_ok=True)
    
    # Load pre-computed representations
    logging.info(f"Loading representations from {config['representations_npz']}")
    data = np.load(config['representations_npz'])
    representations = data['representations']
    labels = data['labels']
    
    logging.info(f"Loaded representations with shape {representations.shape} and labels with shape {labels.shape}")
    
    # Determine if we're using an sklearn model
    use_sklearn = is_sklearn_model(config['classifier'])
    
    # Prepare to collect results
    all_results = []
    
    for exp_idx in range(config['num_experiments']):
        logging.info(f"Starting experiment {exp_idx+1}/{config['num_experiments']} with classifier {config['classifier']}")
        # random_state = config['random_seed'] + exp_idx
        # 随机生成random_state
        random_state = np.random.randint(0, 10000)
        
        # Create datasets
        train_dataset = EthiopiaRepresentationDataset(
            representations=representations,
            labels=labels,
            class_names=config['class_names'],
            samples_per_class=config['samples_per_class'],
            split='train',
            random_state=random_state
        )
        
        val_dataset = EthiopiaRepresentationDataset(
            representations=representations,
            labels=labels,
            class_names=config['class_names'],
            samples_per_class=config['samples_per_class'],
            split='val',
            random_state=random_state
        )
        
        # Get dataset sizes for reporting
        train_size = len(train_dataset)
        val_size = len(val_dataset)
        
        # Different execution paths for sklearn vs PyTorch models
        if use_sklearn:
            # Get numpy data
            X_train, y_train = train_dataset.get_numpy_data()
            X_val, y_val = val_dataset.get_numpy_data()
            
            # Scale data (for logistic regression)
            if config['classifier'] == 'logistic':
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_val = scaler.transform(X_val)
            
            # Create and train classifier
            input_dim = representations.shape[1]
            num_classes = len(config['class_names'])
            
            clf = get_classifier(
                config['classifier'], 
                input_dim, 
                num_classes, 
                random_state=random_state
            )
            
            logging.info(f"Training {config['classifier']} classifier")
            clf.fit(X_train, y_train)
            
            # Evaluate
            train_preds = clf.predict(X_train)
            val_preds = clf.predict(X_val)
            
            # Calculate metrics
            train_metrics = get_detailed_metrics(y_train, train_preds, config['class_names'])
            val_metrics = get_detailed_metrics(y_val, val_preds, config['class_names'])
            
            logging.info(f"Train Accuracy: {train_metrics['overall_accuracy']:.4f}, F1: {train_metrics['overall_f1']:.4f}, "
                         f"Balanced Acc: {train_metrics['balanced_accuracy']:.4f}")
            logging.info(f"Val Accuracy: {val_metrics['overall_accuracy']:.4f}, F1: {val_metrics['overall_f1']:.4f}, "
                         f"Balanced Acc: {val_metrics['balanced_accuracy']:.4f}")
            
            # Generate classification report and confusion matrix
            report = classification_report(y_val, val_preds, target_names=config['class_names'], digits=4)
            conf_mat = confusion_matrix(y_val, val_preds)
            
            logging.info(f"Classification report:\n{report}")
            cm_df = pd.DataFrame(conf_mat, index=config['class_names'], columns=config['class_names'])
            logging.info(f"\nConfusion matrix:\n{cm_df}")
            
            # Store result
            result = {
                'experiment_id': exp_idx + 1,
                'sample_per_pixel': config['samples_per_class'],
                'train_size': train_size,
                'test_size': 0,  # No separate test set
                'val_size': val_size
            }
            result.update(val_metrics)  # Add all metrics
            all_results.append(result)
            
            # Save results to text file
            results_path = os.path.join(config['output_dir'], "logs", 
                                    f"ethiopia_crop_{config['classifier']}_exp{exp_idx+1}_results.txt")
            with open(results_path, "w") as f:
                f.write(f"Classifier: {config['classifier']}\n")
                f.write(f"Train Accuracy: {train_metrics['overall_accuracy']:.4f}, F1: {train_metrics['overall_f1']:.4f}, "
                        f"Balanced Acc: {train_metrics['balanced_accuracy']:.4f}\n")
                f.write(f"Val Accuracy: {val_metrics['overall_accuracy']:.4f}, F1: {val_metrics['overall_f1']:.4f}, "
                        f"Balanced Acc: {val_metrics['balanced_accuracy']:.4f}\n")
                f.write(f"Classification report:\n{report}\n")
                f.write(f"\nConfusion matrix:\n{cm_df}\n")
            
        else:  # PyTorch model
            # Create data loaders for PyTorch
            train_loader = DataLoader(
                train_dataset, 
                batch_size=config['batch_size'], 
                shuffle=True,
                num_workers=config['num_workers'],
                pin_memory=True
            )
            
            val_loader = DataLoader(
                val_dataset, 
                batch_size=config['batch_size'],
                shuffle=False,
                num_workers=config['num_workers']
            )
            
            # Build classification model
            num_classes = len(config['class_names'])
            input_dim = representations.shape[1]
            
            clf_head = LinearProbeHead(input_dim=input_dim, num_classes=num_classes).to(device)
            
            # Optimizer and loss function
            optimizer = AdamW(
                clf_head.parameters(),
                lr=config['lr'], 
                weight_decay=config['weight_decay']
            )
            criterion = nn.CrossEntropyLoss().to(device)
            
            # Training loop
            best_val_acc = 0.0
            best_val_f1 = 0.0
            best_epoch = 0
            checkpoint_path = os.path.join(config['output_dir'], "checkpoints", 
                                         f"ethiopia_crop_{config['classifier']}_exp{exp_idx+1}.pt")
            
            for epoch in range(config['epochs']):
                # Train
                clf_head.train()
                train_preds, train_targets = [], []
                train_loss_sum = 0.0
                
                for repr_batch, labels_batch in tqdm.tqdm(train_loader, 
                                            desc=f"Exp {exp_idx+1}, Epoch {epoch+1}/{config['epochs']} Train"):
                    repr_batch = repr_batch.to(device)
                    labels_batch = labels_batch.to(device)
                    
                    optimizer.zero_grad()
                    logits = clf_head(repr_batch)
                    loss = criterion(logits, labels_batch)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss_sum += loss.item() * repr_batch.size(0)
                    train_preds.extend(logits.argmax(dim=1).cpu().numpy())
                    train_targets.extend(labels_batch.cpu().numpy())
                
                # Training metrics
                avg_train_loss = train_loss_sum / len(train_dataset)
                train_metrics = get_detailed_metrics(train_targets, train_preds, config['class_names'])
                
                logging.info(f"[Exp {exp_idx+1}, Epoch {epoch+1}] Train Loss={avg_train_loss:.4f}, "
                             f"Acc={train_metrics['overall_accuracy']:.4f}, F1={train_metrics['overall_f1']:.4f}, "
                             f"Balanced Acc={train_metrics['balanced_accuracy']:.4f}")
                
                if use_wandb:
                    wandb.log({
                        f"exp{exp_idx+1}/epoch": epoch+1,
                        f"exp{exp_idx+1}/train_loss": avg_train_loss,
                        f"exp{exp_idx+1}/train_acc": train_metrics['overall_accuracy'],
                        f"exp{exp_idx+1}/train_f1": train_metrics['overall_f1'],
                        f"exp{exp_idx+1}/train_balanced_acc": train_metrics['balanced_accuracy']
                    }, step=epoch)
                
                # Validation
                clf_head.eval()
                val_preds, val_targets = [], []
                
                with torch.no_grad():
                    for repr_batch, labels_batch in tqdm.tqdm(val_loader, desc=f"Exp {exp_idx+1}, Validation"):
                        repr_batch = repr_batch.to(device)
                        labels_batch = labels_batch.to(device)
                        
                        logits = clf_head(repr_batch)
                        val_preds.extend(logits.argmax(dim=1).cpu().numpy())
                        val_targets.extend(labels_batch.cpu().numpy())
                
                # Validation metrics
                val_metrics = get_detailed_metrics(val_targets, val_preds, config['class_names'])
                
                logging.info(f"[Exp {exp_idx+1}, Epoch {epoch+1}] Val Acc={val_metrics['overall_accuracy']:.4f}, "
                             f"F1={val_metrics['overall_f1']:.4f}, Balanced Acc={val_metrics['balanced_accuracy']:.4f}")
                
                if use_wandb:
                    wandb.log({
                        f"exp{exp_idx+1}/epoch": epoch+1,
                        f"exp{exp_idx+1}/val_acc": val_metrics['overall_accuracy'],
                        f"exp{exp_idx+1}/val_f1": val_metrics['overall_f1'],
                        f"exp{exp_idx+1}/val_balanced_acc": val_metrics['balanced_accuracy']
                    }, step=epoch)
                
                # Save best model
                if val_metrics['overall_f1'] > best_val_f1:
                    best_val_f1 = val_metrics['overall_f1']
                    best_val_acc = val_metrics['overall_accuracy']
                    best_epoch = epoch+1
                    torch.save(clf_head.state_dict(), checkpoint_path)
                    logging.info(f"Saved best model at epoch {best_epoch}")
            
            # Load best model for final evaluation
            logging.info(f"Loading best checkpoint from epoch {best_epoch} for final evaluation")
            clf_head.load_state_dict(torch.load(checkpoint_path))
            clf_head.eval()
            
            val_preds, val_targets = [], []
            
            with torch.no_grad():
                for repr_batch, labels_batch in tqdm.tqdm(val_loader, desc=f"Exp {exp_idx+1}, Final Evaluation"):
                    repr_batch = repr_batch.to(device)
                    labels_batch = labels_batch.to(device)
                    
                    logits = clf_head(repr_batch)
                    val_preds.extend(logits.argmax(dim=1).cpu().numpy())
                    val_targets.extend(labels_batch.cpu().numpy())
            
            # Calculate final metrics
            val_metrics = get_detailed_metrics(val_targets, val_preds, config['class_names'])
            report = classification_report(val_targets, val_preds, target_names=config['class_names'], digits=4)
            conf_mat = confusion_matrix(val_targets, val_preds)
            
            logging.info(f"Experiment {exp_idx+1} final validation Acc: {val_metrics['overall_accuracy']:.4f}, "
                         f"F1: {val_metrics['overall_f1']:.4f}, Balanced Acc: {val_metrics['balanced_accuracy']:.4f}")
            logging.info(f"Classification report:\n{report}")
            
            cm_df = pd.DataFrame(conf_mat, index=config['class_names'], columns=config['class_names'])
            logging.info(f"\nConfusion matrix:\n{cm_df}")
            
            # Save results to text file
            results_path = os.path.join(config['output_dir'], "logs", 
                                    f"ethiopia_crop_{config['classifier']}_exp{exp_idx+1}_results.txt")
            with open(results_path, "w") as f:
                f.write(f"Best epoch: {best_epoch}\n")
                f.write(f"Final validation Acc: {val_metrics['overall_accuracy']:.4f}\n")
                f.write(f"Final validation F1: {val_metrics['overall_f1']:.4f}\n")
                f.write(f"Final validation Balanced Acc: {val_metrics['balanced_accuracy']:.4f}\n")
                f.write(f"Classification report:\n{report}\n")
                f.write(f"\nConfusion matrix:\n{cm_df}\n")
            
            # Store result
            result = {
                'experiment_id': exp_idx + 1,
                'sample_per_pixel': config['samples_per_class'],
                'train_size': train_size,
                'test_size': 0,  # No separate test set
                'val_size': val_size
            }
            result.update(val_metrics)  # Add all metrics
            all_results.append(result)
    
    # Aggregate results across experiments
    df_results = pd.DataFrame(all_results)
    
    # Calculate mean and std for key metrics
    mean_acc = df_results['overall_accuracy'].mean()
    std_acc = df_results['overall_accuracy'].std()
    mean_f1 = df_results['overall_f1'].mean()
    std_f1 = df_results['overall_f1'].std()
    mean_balanced_acc = df_results['balanced_accuracy'].mean()  # Added mean balanced accuracy
    std_balanced_acc = df_results['balanced_accuracy'].std()    # Added std of balanced accuracy
    
    logging.info(f"===== Summary for {config['num_experiments']} experiments "
                 f"with {config['samples_per_class']} samples per class using {config['classifier']} =====")
    logging.info(f"Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    logging.info(f"Mean F1 Score: {mean_f1:.4f} ± {std_f1:.4f}")
    logging.info(f"Mean Balanced Accuracy: {mean_balanced_acc:.4f} ± {std_balanced_acc:.4f}")  # Log balanced accuracy
    
    # Save summary to text file
    summary_path = os.path.join(config['output_dir'], "logs", 
                            f"ethiopia_crop_{config['classifier']}_summary_{config['samples_per_class']}samples.txt")
    with open(summary_path, "w") as f:
        f.write(f"===== Summary for {config['num_experiments']} experiments "
                f"with {config['samples_per_class']} samples per class using {config['classifier']} =====\n")
        f.write(f"Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}\n")
        f.write(f"Mean F1 Score: {mean_f1:.4f} ± {std_f1:.4f}\n")
        f.write(f"Mean Balanced Accuracy: {mean_balanced_acc:.4f} ± {std_balanced_acc:.4f}\n\n")
        f.write("Individual experiment results:\n")
        f.write(df_results.to_string())
    
    # Save detailed results to CSV
    csv_path = os.path.join(config['output_dir'], 
                          f"{config['samples_per_class']}_{config['num_experiments']}_{config['classifier']}.csv")
    df_results.to_csv(csv_path, index=False)
    logging.info(f"Detailed results saved to {csv_path}")
    
    if use_wandb:
        wandb.log({
            "summary/mean_acc": mean_acc,
            "summary/std_acc": std_acc,
            "summary/mean_f1": mean_f1,
            "summary/std_f1": std_f1,
            "summary/mean_balanced_acc": mean_balanced_acc,
            "summary/std_balanced_acc": std_balanced_acc,
            "summary/samples_per_class": config['samples_per_class'],
            "summary/classifier": config['classifier']
        })
        wandb_run.finish()

if __name__ == "__main__":
    main()