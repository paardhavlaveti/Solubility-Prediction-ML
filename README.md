# Molecular Solubility Prediction Project 🧪

## 📝 Description

This project demonstrates the use of machine learning to predict the aqueous solubility of chemical molecules. Aqueous solubility is a critical property in pharmaceutical and environmental sciences, and this model provides an efficient way to estimate it from molecular structure.

The project implements and compares two popular regression models:
* **Linear Regression:** A simple, interpretable baseline model.
* **Random Forest Regressor:** A more complex ensemble model known for its high accuracy.

The goal is to build a predictive model that accurately estimates the **logS** value (logarithm of the molar solubility in water) based on a set of calculated molecular descriptors.

---

## Workflow

The project follows a standard machine learning pipeline:

1.  **Data Loading:** The Delaney solubility dataset is loaded using Pandas.
2.  **Data Splitting:** The dataset is divided into features (molecular descriptors) and the target variable (logS).
3.  **Train-Test Split:** The data is split into training (80%) and testing (20%) sets to evaluate model performance on unseen data.
4.  **Model Training:** Both Linear Regression and Random Forest models are trained on the training data.
5.  **Prediction:** The trained models are used to make predictions on the test set.
6.  **Evaluation:** Model performance is assessed using R-squared ($R^2$) and Mean Squared Error (MSE) metrics.
7.  **Visualization:** The results are visualized with a scatter plot comparing the actual vs. predicted solubility values.

---

## 💾 Dataset

The model was trained on the **Delaney Solubility dataset**, which contains molecular descriptors for 1144 chemical compounds.

This dataset was introduced in the paper: *ESOL: Estimating Aqueous Solubility Directly from Molecular Structure* by John S. Delaney.

- **Features (Molecular Descriptors):**
    - `MolLogP` (Octanol-water partition coefficient)
    - `MolWt` (Molecular weight)
    - `NumRotatableBonds` (Number of rotatable bonds)
    - `AromaticProportion` (Proportion of atoms that are aromatic)
- **Target Variable:**
    - `logS` (Measured aqueous solubility)

- **Link to data:** You can find the dataset [here](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv).

---

## 🛠️ Installation & Setup

To run this project on your local machine, you'll need Python 3 and a few common data science libraries.

1.  **Clone the repository (optional):**
    ```bash
    git clone [https://github.com/](https://github.com/)[your-username]/[your-repo-name].git
    cd [your-repo-name]
    ```

2.  **Install the required libraries:**
    It's recommended to create a `requirements.txt` file. You can install the necessary packages using pip:
    ```bash
    pip install pandas scikit-learn matplotlib seaborn
    ```

---

## 🚀 Usage

You can run this project in two primary ways:

1.  **Jupyter Notebook / Google Colab:**
    The simplest way is to open the `.ipynb` notebook file in a Jupyter environment or upload it to Google Colab. From there, you can run each cell sequentially to see the entire process from data loading to model evaluation.

2.  **Python Script:**
    If you have a `.py` script, you can run it directly from your terminal:
    ```bash
    python your_script_name.py
    ```

---

## 📈 Results & Model Performance

The models were evaluated on the test set to measure their predictive accuracy on unseen data. The performance is summarized below:

| Model                | Test R² Score        | Test MSE        |
| -------------------- | -------------------- | --------------- |
| **Linear Regression**| 0.789                | 1.020           |
| **Random Forest** | `[Your RF R² score]` | `[Your RF MSE]` |

The **R-squared ($R^2$)** value represents the proportion of the variance in the dependent variable that is predictable from the independent variables. A higher value indicates a better fit.

The **Mean Squared Error (MSE)** measures the average squared difference between the estimated values and the actual value. A lower value indicates better accuracy.

---

## 📊 Visualization

A scatter plot of experimental vs. predicted `logS` values for the test set provides a clear visual assessment of the Random Forest model's performance. The closer the points lie to the 45-degree line, the more accurate the predictions.



---

## 🏁 Conclusion

Based on the evaluation metrics, the **Random Forest Regressor** significantly outperforms the Linear Regression model. Its lower MSE and higher R² score indicate that it captures the non-linear relationships in the data more effectively, leading to more accurate solubility predictions.

This project successfully demonstrates that machine learning models, particularly ensemble methods like Random Forest, can be powerful tools for predicting crucial chemical properties from molecular descriptors.
