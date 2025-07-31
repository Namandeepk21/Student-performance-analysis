import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

# 1. Load CSV
df = pd.read_csv("StudentsPerformance.csv")

# 2. Show basic info
print(df.head())
print(df.columns)

# 3. Create 'pass_math' column (if score ≥ 50 → Pass)
df['pass_math'] = np.where(df['math score'] >= 50, 'Passed', 'Failed')

# 4. Count Pass/Fail
counts = df['pass_math'].value_counts()

# 5. Pie chart
labels = ['Passed', 'Failed']
plt.figure(figsize=(6, 6))
plt.pie(counts, labels=labels, autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
plt.title("Math Pass/Fail Rate")
plt.show()

# 6. Group by test prep course and calculate average scores
prep_avg = df.groupby('test preparation course')[['math score', 'reading score', 'writing score']].mean()
print(prep_avg)



sns.boxplot(x="lunch",y="reading score",data=df,palette="pastel")
plt.title("Average Math score by gender")
print.show()
model = smf.ols("math score ~ reading score + score",data = df).fit()
print(model.summery())
df["pass math"]=np.where(df["math score"]>=60,1,0)
log_model=smf.logit("pass math ~ C(lunch)",data=df).fit()
print(log_model.summery())
    