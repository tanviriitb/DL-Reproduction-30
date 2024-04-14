import pandas as pd
from sklearn.model_selection import train_test_split

def create_new_datasets(base_train_path, base_test_path, base_val_path):
    # Load the datasets
    base_train = pd.read_csv(base_train_path)
    base_test = pd.read_csv(base_test_path)
    base_val = pd.read_csv(base_val_path)
    
    # Step 1: Concatenate the training, testing, and validation sets
    full_training_set = pd.concat([base_train, base_test, base_val])
    
    # Initialize containers for the new datasets
    new_train_df = pd.DataFrame(columns=full_training_set.columns)
    new_test_df = pd.DataFrame(columns=full_training_set.columns)
    new_val_df = pd.DataFrame(columns=full_training_set.columns)

    # Step 2: Select new testing and validation sets
       # Process each language separately to maintain proportion
    for language in full_training_set['LANGUAGE'].unique():
        language_data = full_training_set[full_training_set['LANGUAGE'] == language]
        
        # For each word in the language
        for word in language_data['WORD'].unique():
            word_data = language_data[language_data['WORD'] == word]
            
            # Randomly select 100 samples for testing, 100 for validation, rest for training
            remaining, test_samples = train_test_split(word_data, test_size=100, random_state=42)
            train_samples, val_samples = train_test_split(remaining, test_size=100, random_state=42)
            
            # Append to their respective DataFrames
            new_train_df = pd.concat([new_train_df, train_samples])
            new_test_df = pd.concat([new_test_df, test_samples])
            new_val_df = pd.concat([new_val_df, val_samples])
    
    # Order by language (and word if needed)
    # new_train_df = new_train_df.sort_values(by=['LANGUAGE', 'WORD'])
    # new_test_df = new_test_df.sort_values(by=['LANGUAGE', 'WORD'])
    # new_val_df = new_val_df.sort_values(by=['LANGUAGE', 'WORD'])

    # Save the new datasets
    new_train_df.to_csv("base_train_1.csv", index=False)
    new_test_df.to_csv("base_test_1.csv", index=False)
    new_val_df.to_csv("base_val_1.csv", index=False)
    
    print("New datasets have been saved.")

if __name__ == "__main__":
    base_train_path = 'base_train_initial.csv'
    base_test_path = 'base_test_initial.csv'
    base_val_path = 'base_val_initial.csv'
    
    
    create_new_datasets(base_train_path, base_test_path, base_val_path)

