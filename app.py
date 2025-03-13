import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import gradio as gr

# Load and preprocess the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 
           'marital-status', 'occupation', 'relationship', 'race', 
           'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'target']
df = pd.read_csv(url, header=None, names=columns, na_values=' ?', skipinitialspace=True)
df.dropna(inplace=True)
df['target'] = df['target'].apply(lambda x: 1 if x == '>50K' else 0)
df = pd.get_dummies(df, columns=df.select_dtypes(include=['object']).columns, drop_first=True)

# Split the dataset
X = df.drop('target', axis=1)
y = df['target']

# Train the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Function to make predictions
def predict_income(age, workclass, fnlwgt, education, education_num, marital_status, occupation, relationship, race, sex, capital_gain, capital_loss, hours_per_week, native_country):
    features = {
        "age": age,
        "workclass_" + workclass: 1,
        "fnlwgt": fnlwgt,
        "education_" + education: 1,
        "education-num": education_num,
        "marital-status_" + marital_status: 1,
        "occupation_" + occupation: 1,
        "relationship_" + relationship: 1,
        "race_" + race: 1,
        "sex_" + sex: 1,
        "capital-gain": capital_gain,
        "capital-loss": capital_loss,
        "hours-per-week": hours_per_week,
        "native-country_" + native_country: 1
    }
    
    # Create a DataFrame for the input features
    input_data = pd.DataFrame([features], columns=X.columns)
    prediction = model.predict(input_data)
    
    return "Income > $50K" if prediction[0] == 1 else "Income <= $50K"

# Define the input features
input_features = [
    gr.Slider(minimum=0, maximum=100, label="Age"),
    gr.Dropdown(choices=["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", 
                         "Local-gov", "State-gov", "Without-pay", "Never-worked"], label="Work Class"),
    gr.Number(label="Final Weight"),
    gr.Dropdown(choices=["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", 
                         "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "6th", 
                         "5th", "4th", "3rd", "2nd", "1st", "Preschool"], label="Education"),
    gr.Slider(minimum=1, maximum=16, label="Education Number"),
    gr.Dropdown(choices=["Married-civ-spouse", "Divorced", "Never-married", "Separated", 
                         "Widowed", "Married-spouse-absent", "Married-AF-spouse"], label="Marital Status"),
    gr.Dropdown(choices=["Tech-support", "Craft-repair", "Other-service", "Sales", 
                         "Exec-managerial", "Prof-specialty", "Handlers-cleaners", 
                         "Machine-op-inspct", "Adm-clerical", "Farming-fishing", 
                         "Transport-moving", "Priv-house-serv", "Protective-serv", 
                         "Armed-Forces"], label="Occupation"),
    gr.Dropdown(choices=["Wife", "Own-child", "Husband", "Not-in-family", 
                         "Other-relative", "Unmarried"], label="Relationship"),
    gr.Dropdown(choices=["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", 
                         "Other", "Black"], label=" Race"),
    gr.Dropdown(choices=["Female", "Male"], label="Sex"),
    gr.Slider(minimum=0, maximum=100000, label="Capital Gain"),
    gr.Slider(minimum=0, maximum=5000, label="Capital Loss"),
    gr.Slider(minimum=1, maximum=99, label="Hours per Week"),
    gr.Dropdown(choices=["United-States", "Cambodia", "England", "Puerto-Rico", 
                         "Canada", "Germany", "Outlying-US(Guam-USVI-etc)", 
                         "India", "Japan", "Greece", "South", "China", 
                         "Cuba", "Iran", "Honduras", "Philippines", "Italy", 
                         "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal", 
                         "Ireland", "France", "Dominican-Republic", "Laos", 
                         "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", 
                         "Guatemala", "Nicaragua", "Scotland", "Thailand", 
                         "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", 
                         "Hong", "Holand-Netherlands"], label="Native Country")
]

# Define the output
output = gr.Textbox(label="Income Prediction") # Change gr.outputs.Textbox to gr.Textbox

# Create the Gradio interface
gr.Interface(fn=predict_income, inputs=input_features, outputs=output, 
             title="Income Analysis Using Machine Learning Algorithm", description="Your Future Analyzed:Income Prediction Made easier").launch()
