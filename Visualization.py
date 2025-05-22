import pandas as pd
import matplotlib.pyplot as plt

imp = pd.read_csv("feature_importance.csv")
imp.plot.barh(x="feature", y="importance", legend=False)
plt.xlabel("Mean Permutation Importance")
plt.title("LightGBM Feature Importances")
plt.tight_layout()
plt.savefig("feature_importance.png")
