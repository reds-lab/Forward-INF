import numpy as np
import torch
import time
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from joblib import dump, load
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import argparse
import random

def set_seed(seed):
    # Set a seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed) 
    # If you are using CUDA:
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
def fetch_dataset(dataset_name, seed):
    if dataset_name == 'cifar10':
        dataset = fetch_openml('CIFAR_10', version=1)
        dataset.data = dataset.data.astype(np.float32) / 255.0
        train_size = 0.08 # 0.08
    elif dataset_name == 'fmnist':
        dataset = fetch_openml('Fashion-MNIST', version=1)
        dataset.data = dataset.data.astype(np.float32) / 255.0
        train_size = 0.07 # 0.07
    else:  # Default to MNIST
        dataset = fetch_openml('mnist_784', version=1)
        dataset.data = dataset.data.astype(np.float32) / 255.0
        train_size = 0.07 # 0.07
    
    # Split the dataset
    validation_size = 5000
    return train_test_split(dataset.data, dataset.target, train_size=train_size, test_size=validation_size, random_state=seed)

def utility_fcn(x_train, y_train, x_test, y_test, indicator, idx, sel_idx, batch_size, weight_decay, max_iter, remove, repeat, dataset, seed):
    set_seed(seed)    
    
    model_path = f'./saved_models/tst/{dataset}_{remove}' # saved_models
    model_filename = f'{model_path}/logisticRegr_model_{indicator}{idx}.joblib'

    if repeat == 0:
        if indicator == 'dual_loo_pre_': # load the model if it's pre-trained , [ DualLOO first term == LOO second term , do not depend on new test point ]
            model_filename = f'{model_path}/logisticRegr_model_loo_total_{idx}.joblib'
            logisticRegr = load(model_filename)
        else:  # Train the model otherwise , [ DualLOO second , with new test point, DualLOO second will be different, so no need to save it ]
            logisticRegr = LogisticRegression(
                C=1.0 / (len(x_train) * weight_decay),        
                solver='lbfgs',
                multi_class='multinomial',
                warm_start=True,
                max_iter=max_iter)
            logisticRegr.fit(x_train, y_train)
            
            # Save the model for later use if required , [ LOO terms do not depend on new test point, so we need to save it ]
            if indicator in ['loo_total_', 'loo_pre_']:
                dump(logisticRegr, model_filename)
                
    elif repeat == 1:
        if indicator == 'dual_loo_pre_':  # Load the model if it's pre-trained [ do not depend on new test point, so just load it ]
            model_filename = f'{model_path}/logisticRegr_model_loo_total_{idx}.joblib'
            logisticRegr = load(model_filename)        
        elif indicator in ['loo_total_', 'loo_pre_']: # [ do not depend on new test point , so just load it ]
            logisticRegr = load(model_filename) 
        else:  # Train the model for each test point [ DualLOO second ]
            logisticRegr = LogisticRegression(
                C=1.0 / (len(x_train) * weight_decay),        
                solver='lbfgs',
                multi_class='multinomial',
                warm_start=True,
                max_iter=max_iter)
            logisticRegr.fit(x_train, y_train)
        
    validation_data_index = 0  # Change this to the desired index
    probs = logisticRegr.predict_proba([x_test])
    test_loss = -np.log(probs[0, y_test])

#     # Save the loss         
#     score_path = f'./saved_scores/{dataset}_{remove}'
#     score_filename = f'{score_path}/logisticRegr_model_{indicator}{idx}_{sel_idx}.pt'
#     torch.save(test_loss.item(), score_filename)  
    
    return test_loss.item()

def main(args):
    set_seed(args.seed)
    train_images, validation_images, train_labels, validation_labels = fetch_dataset(args.dataset, args.seed)
    
    # Convert pandas dataframes to numpy arrays
    train_img_np = train_images.to_numpy().astype(np.float32)
    test_img_np = validation_images.to_numpy().astype(np.float32)
    train_lb_np = train_labels.to_numpy().astype(np.long)
    test_lb_np = validation_labels.to_numpy().astype(np.long)
    
    seed = args.seed
    dataset = args.dataset
    batch_size = args.batch_size
    weight_decay = args.weight_decay
    max_iter = args.max_iter
    remove = args.remove ## addone or removeone    

    # Train Logistic Regression model using scikit-learn on the sampled data
    start_time = time.time()
    logisticRegr = LogisticRegression(
        C=1.0 / (len(train_images) * weight_decay),
        solver='lbfgs',
        multi_class='multinomial',
        warm_start=True,
        max_iter=max_iter)
    logisticRegr.fit(train_images, train_labels)
    print(f"Training elapsed time: {time.time() - start_time} seconds")    
    
    # Predict on training set
    train_predictions = logisticRegr.predict(train_images)
    train_accuracy = (train_predictions == train_labels).mean()
    print("Training accuracy:", train_accuracy)

    # Predict on validation set
    validation_predictions = logisticRegr.predict(validation_images)
    validation_accuracy = (validation_predictions == validation_labels).mean()
    print("Validation accuracy:", validation_accuracy)
    
    losses = []
    for i in range(len(test_img_np)):
        probs = logisticRegr.predict_proba([test_img_np[i]])
        test_loss = -np.log(probs[0, test_lb_np[i]])
        losses.append(test_loss)

    top_val, top_indices = torch.sort(torch.from_numpy(np.array(losses)), descending=True)
    top_10_val, top_10_indices = top_val[:10], top_indices[:10] 
    
    print(f'Top 10 highest loss indices: {top_10_indices}')
    print(f'Top 10 highest losses: {top_10_val}')

    # # LOO vs Dual
    # score_path = f'./temp/scores/{dataset}_{remove}/ours_indices_' + dataset + '.npy' ## saved_scores/
    # np.save(score_path, top_10_indices)

    cor_score_lst = []

    N = len(train_img_np)
    cnt_ = 0
    for sel_ind in top_10_indices.tolist():
        #-------------------------------------------------#    
        if cnt_ == 0:
            repeat = 0
        else:
            repeat = 1
        #-------------------------------------------------#
        ### LOO
        loo_score = []
        pre_loss_lst = []
        total_loss_lst = []   
        cnt = 0
        
        ## x_test
        x_test, y_test = test_img_np[sel_ind], test_lb_np[sel_ind]    

        total_loss = utility_fcn(train_img_np, train_lb_np, x_test, y_test, 'loo_total_', N, sel_ind, batch_size, weight_decay, max_iter, remove, repeat, dataset, seed)

        start_time = time.time()
        for idx in range(N): ## training data samples, and test data sample is fixed 

            if remove == "addone":
                x_train_addsone, y_train_addsone = np.concatenate((train_img_np, train_img_np[idx:idx+1])), np.concatenate((train_lb_np, train_lb_np[idx:idx+1])) ## Concatenating the ith training data point to the original training data
                pre_loss = utility_fcn(x_train_addsone, y_train_addsone, x_test, y_test, 'loo_addone_', idx, sel_ind, batch_size, weight_decay, max_iter, remove, repeat, dataset, seed)

                loo_score.append(total_loss-pre_loss) ## (D U i) - (D) 
                total_loss_lst.append(total_loss)
                pre_loss_lst.append(pre_loss)
                
            else:
                loo_index = torch.ones(N, dtype=bool)
                loo_index[idx] = 0   

                x_train_loo, y_train_loo = train_img_np[loo_index], train_lb_np[loo_index] ## D_\i on x_test        
                pre_loss = utility_fcn(x_train_loo, y_train_loo, x_test, y_test, 'loo_pre_', idx, sel_ind, batch_size, weight_decay, max_iter, remove, repeat, dataset, seed)

                loo_score.append(pre_loss-total_loss) ## (D_\i U i) - D
                total_loss_lst.append(total_loss)
                pre_loss_lst.append(pre_loss)                
            cnt += 1

            if cnt%50 == 0:
                print(f" current idx is {cnt} ")

        print(f" time elapsed : {time.time() - start_time}")

        ###################################################
        ### Dual LOO
        dual_loo_score = []
        pre_train_loss_lst = []
        total_train_loss_lst = []        
        cnt = 0

        start_time = time.time()
        for idx in range(N): ## training data samples, and test data sample is fixed 

            ## x_i sample
            x_i, y_i = train_img_np[idx], train_lb_np[idx]    

            ## x_test
            x_test, y_test = test_img_np[sel_ind:sel_ind+1], test_lb_np[sel_ind:sel_ind+1]

            ## D on x_i     
            pre_train_loss = utility_fcn(train_img_np, train_lb_np, x_i, y_i, 'dual_loo_pre_', N, sel_ind, batch_size, weight_decay, max_iter, remove, repeat, dataset, seed)

            ## (D U x_test) on x_i
            x_train_loo, y_train_loo = np.concatenate([train_img_np, x_test]), np.concatenate([train_lb_np, y_test])
            total_train_loss = utility_fcn(x_train_loo, y_train_loo, x_i, y_i, 'dual_loo_total_', idx, sel_ind, batch_size, weight_decay, max_iter, remove, repeat, dataset, seed)
            dual_loo_score.append(pre_train_loss-total_train_loss) ## D - (D U x_test) / it was total_train_loss-pre_train_loss before,            
            total_train_loss_lst.append(total_train_loss)            
            pre_train_loss_lst.append(pre_train_loss)
            cnt += 1

            if cnt%50 == 0:
                print(f" current idx is {cnt} ")

        print(f" time elapsed : {time.time() - start_time}")
 
        ################################################### 
        # Save the arrays to the desired folder
        np.save('./saved_scores/' + dataset + '_' + remove + '/dual_loo_score_1st_' + str(sel_ind) +'.npy', pre_train_loss) # saved_scores
        np.save('./saved_scores/' + dataset + '_' + remove + '/dual_loo_score_2nd_' + str(sel_ind) +'.npy', total_train_loss)        
        np.save('./saved_scores/' + dataset + '_' + remove + '/loo_score_1st_' + str(sel_ind) +'.npy', pre_loss)
        np.save('./saved_scores/' + dataset + '_' + remove + '/loo_score_2nd_' + str(sel_ind) +'.npy', total_loss)
        
        np.save('./saved_scores/' + dataset + '_' + remove + '/dual_loo_score_' + str(sel_ind) +'.npy', dual_loo_score)
        np.save('./saved_scores/' + dataset + '_' + remove + '/loo_score_' + str(sel_ind) +'.npy', loo_score)
        
        ###################################################
        ### Loo scores vs LOO scores
        # Convert PyTorch tensors to NumPy arrays
        array1 = np.array(dual_loo_score)
        array2 = np.array(loo_score)

        # Calculate the correlation coefficient
        correlation_coefficient, p = pearsonr(array1, array2)

        print("Correlation Coefficient:", correlation_coefficient)
        cor_score_lst.append(correlation_coefficient)
        cnt_ += 1
        print("cor_score_lst:", cor_score_lst)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to run logistic regression with different datasets.")
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10', 'fmnist'], help='Dataset to use; mnist, cifar10, or fmnist')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for regularization')
    parser.add_argument('--max_iter', type=int, default=1000, help='Maximum number of iterations for training')
    parser.add_argument('--remove', type=str, default='removeone', choices=['addone', 'removeone'], help='Method to remove data points')

    args = parser.parse_args()
    main(args)