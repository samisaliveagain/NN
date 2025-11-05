import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(filepath="data/Concrete_Data.xls"):
    df = pd.read_csv(filepath)
    df.columns = [
        "Cement", "Blast_Furnace_Slag", "Fly_Ash", "Water",
        "Superplasticizer", "Coarse_Aggregate", "Fine_Aggregate",
        "Age", "Strength"
    ]
    
    X = df.drop(columns=["Strength"]).values
    y = df["Strength"].values.reshape(-1, 1)

    # Normalize for stable NN training
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, scaler_y
