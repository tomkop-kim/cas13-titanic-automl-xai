import pandas as pd

def preprocess_titanic(train: pd.DataFrame, test: pd.DataFrame):
    data = train.copy()
    test_proc = test.copy()

    age_median = data["Age"].median()
    embarked_mode = data["Embarked"].mode()[0]

    drop_cols = ["PassengerId", "Name", "Ticket", "Cabin"]
    data = data.drop(columns=drop_cols, errors="ignore")
    test_proc = test_proc.drop(columns=drop_cols, errors="ignore")

    data["Age"] = data["Age"].fillna(age_median)
    data["Embarked"] = data["Embarked"].fillna(embarked_mode)

    test_proc["Age"] = test_proc["Age"].fillna(age_median)
    test_proc["Embarked"] = test_proc["Embarked"].fillna(embarked_mode)

    data["Sex"] = data["Sex"].map({"male": 0, "female": 1})
    test_proc["Sex"] = test_proc["Sex"].map({"male": 0, "female": 1})

    data = pd.get_dummies(data, columns=["Embarked"], drop_first=True)
    test_proc = pd.get_dummies(test_proc, columns=["Embarked"], drop_first=True)

    train_cols = [c for c in data.columns if c != "Survived"]
    test_proc = test_proc.reindex(columns=train_cols, fill_value=0)

    return data, test_proc