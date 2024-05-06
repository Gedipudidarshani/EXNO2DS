# EXNO2DS
# AIM:To perform Exploratory Data Analysis on the given data set.
      
# EXPLANATION:
  The primary aim with exploratory analysis is to examine the data for distribution, outliers and anomalies to direct specific testing of your hypothesis.
  
# ALGORITHM:
STEP 1: Import the required packages to perform Data Cleansing,Removing Outliers and Exploratory Data Analysis.

STEP 2: Replace the null value using any one of the method from mode,median and mean based on the dataset available.

STEP 3: Use boxplot method to analyze the outliers of the given dataset.

STEP 4: Remove the outliers using Inter Quantile Range method.

STEP 5: Use Countplot method to analyze in a graphical method for categorical data.

STEP 6: Use displot method to represent the univariate distribution of data.

STEP 7: Use cross tabulation method to quantitatively analyze the relationship between multiple variables.

STEP 8: Use heatmap method of representation to show relationships between two variables, one plotted on each axis.

## CODING AND OUTPUT
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
dt=pd.read_csv("/titanic_dataset.csv")
dt
```
![Screenshot 2024-03-14 134555](https://github.com/Munimadhuriganji/EXNO2DS/assets/138849444/180f8314-4da2-450a-8d99-eef9fee537d7)

```
dt.info()
```
![Screenshot 2024-03-14 134605](https://github.com/Munimadhuriganji/EXNO2DS/assets/138849444/fffb0bc9-dc68-4da5-bf6d-101caf0530bc)

```
dt.shape
```
![Screenshot 2024-03-14 134612](https://github.com/Munimadhuriganji/EXNO2DS/assets/138849444/97c00ed0-327f-4056-9d22-28b03ff72fd2)


```
dt.set_index("PassengerId",inplace=True)
dt.describe()
```
![Screenshot 2024-03-14 134618](https://github.com/Munimadhuriganji/EXNO2DS/assets/138849444/88e06830-353b-4f25-b7eb-d84c9f321892)


```
dt.nunique()
```
![Screenshot 2024-03-14 134628](https://github.com/Munimadhuriganji/EXNO2DS/assets/138849444/21b5abd2-e523-4617-822f-156517ee61f0)

```
dt["Survived"].value_counts()
```
![Screenshot 2024-03-14 134628](https://github.com/Munimadhuriganji/EXNO2DS/assets/138849444/f2626861-2b5b-4e59-a636-c90a756c69b4)

```
per=(dt["Survived"].value_counts()/dt.shape[0]*100).round(2)
per
```
![Screenshot 2024-03-14 134634](https://github.com/Munimadhuriganji/EXNO2DS/assets/138849444/3a9d61ea-3740-4c8a-a690-762ed0b99408)

```
sns.countplot(data=dt,x="Survived")
```
![Screenshot 2024-03-14 134644](https://github.com/Munimadhuriganji/EXNO2DS/assets/138849444/6af37f5c-923e-41e6-8769-8ff009ed7fc5)


```
dt
```

![Screenshot 2024-03-14 134701](https://github.com/Munimadhuriganji/EXNO2DS/assets/138849444/101619d9-ee97-4304-936a-63daa9741f8c)

```
dt.Pclass.unique()
```
![Screenshot 2024-03-14 134710](https://github.com/Munimadhuriganji/EXNO2DS/assets/138849444/7617774f-0284-4e10-beb1-3cf0acfb4e2f)

```
dt.rename(columns={'Sex':'Gender'},inplace=True)
dt
```
![Screenshot 2024-03-14 134724](https://github.com/Munimadhuriganji/EXNO2DS/assets/138849444/c43ba943-6777-4a29-a003-700e0b79e387)

```
sns.catplot(x="Gender",col="Survived",kind="count",data=dt,height=5,aspect=.7)
```
![Screenshot 2024-03-14 134724](https://github.com/Munimadhuriganji/EXNO2DS/assets/138849444/069682c9-916b-4bb9-8752-f5de20be7c28)

```
sns.catplot(x='Survived',hue="Gender",data=dt,kind='count')
```
![Screenshot 2024-03-14 134733](https://github.com/Munimadhuriganji/EXNO2DS/assets/138849444/170a9720-83f7-4d7c-b29b-6b80b8c225e9)

```
dt.boxplot(column="Age",by="Survived")
```
![Screenshot 2024-03-14 134742](https://github.com/Munimadhuriganji/EXNO2DS/assets/138849444/43864174-9226-4f6a-bc8e-82930aee101b)

```
sns.scatterplot(x=dt["Age"],y=dt["Fare"])
```
![Screenshot 2024-03-14 134800](https://github.com/Munimadhuriganji/EXNO2DS/assets/138849444/a7fef944-5fde-4e82-9508-1d3773972b1e)

```
sns.jointplot(x="Age",y="Fare",data=dt)
```
![Screenshot 2024-03-14 134818](https://github.com/Munimadhuriganji/EXNO2DS/assets/138849444/4ef30523-659f-4c15-a276-7b155b959c73)

```
fig,ax1=plt.subplots(figsize=(8,5))
sns.boxplot(ax=ax1,x="Pclass",y="Age",hue="Gender",data=dt)
```
![Screenshot 2024-03-14 134828](https://github.com/Munimadhuriganji/EXNO2DS/assets/138849444/7c45e04f-b65c-4ecb-9a1c-affed168d513)

```
sns.catplot(data=dt,col="Survived",x="Gender",hue="Pclass",kind="count")
```
![Screenshot 2024-03-14 134839](https://github.com/Munimadhuriganji/EXNO2DS/assets/138849444/1f802127-39b8-490f-8ab3-181667dd47e0)

```
corr=dt.corr()
sns.heatmap(corr,annot=True)
```
![Screenshot 2024-03-14 134925](https://github.com/Munimadhuriganji/EXNO2DS/assets/138849444/10671ca6-8b24-4ff9-b73b-1914de7fdca4)

```
sns.pairplot(dt)
```
![Screenshot 2024-03-14 135005](https://github.com/Munimadhuriganji/EXNO2DS/assets/138849444/5a1a4df5-a064-4ccf-991c-c68dfd8c1f79)




# RESULT
  To perform Exploratory Data Analysis on the given data set is succesfully completed.
