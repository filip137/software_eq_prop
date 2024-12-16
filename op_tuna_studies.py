import pandas as pd
from optuna.trial import TrialState

def extract_study_data(study):
    """
    Extract data from an Optuna study into a Pandas DataFrame.
    
    Parameters:
        study (optuna.Study): The Optuna study to extract data from.
    
    Returns:
        pd.DataFrame: A DataFrame containing trial information, including parameters, value, and state.
    """
    # Retrieve all trials
    trials = study.get_trials(deepcopy=False)
    
    # Prepare the data for DataFrame
    data = []
    for trial in trials:
        trial_data = {
            "trial_number": trial.number,
            "state": trial.state.name,
            "value": trial.value,
        }
        # Add parameters
        trial_data.update(trial.params)
        # Add user attributes, if any
        trial_data.update(trial.user_attrs)
        # Add system attributes, if any
        trial_data.update(trial.system_attrs)
        data.append(trial_data)
    
    # Create a DataFrame
    df = pd.DataFrame(data)
    
    return df

import optuna

# Load the study
directory = '/home/filip/optuna'
storage_url = "sqlite:///" + directory + "/optuna_studies.db"
study_name = "eq_prop_memnist"

study = optuna.load_study(study_name=study_name, storage=storage_url)

# Extract the data
study_data = extract_study_data(study)

# Display the DataFrame
print(study_data)

# Optionally, save to CSV for further analysis
study_data.to_csv("optuna_study_data.csv", index=False)