## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
df=pd.read_csv('/content/Encoding Data.csv')
df
```
![image](https://github.com/hindhujanaki/EXNO-3-DS/assets/148514666/87a3296c-1a0c-4e9a-82e1-f1258fed9593)

# Ordinal Encoding
```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![image](https://github.com/hindhujanaki/EXNO-3-DS/assets/148514666/8c57ff2e-30a2-4a7b-8615-d9c7c32193ee)

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![image](https://github.com/hindhujanaki/EXNO-3-DS/assets/148514666/7de784f2-5057-48cb-a406-39e5d55db4a9)

# Label Encoder
```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/hindhujanaki/EXNO-3-DS/assets/148514666/a081d61a-e610-4dbf-944a-4a181c90638c)
```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
```
![image](https://github.com/hindhujanaki/EXNO-3-DS/assets/148514666/e36c0d06-fa40-42c5-aaa9-1d2a41724439)

```
df2=pd.concat([df2,enc],axis=1)
df2
```
![image](https://github.com/hindhujanaki/EXNO-3-DS/assets/148514666/83425941-07ff-44af-91b2-80149b3ac088)
```
pd.get_dummies(df2,columns=["nom_0"])
```
![image](https://github.com/hindhujanaki/EXNO-3-DS/assets/148514666/1d6b19d9-0646-4001-a2a6-84911c68f4fc)

# Binary Encoder
```
pip install --upgrade category_encoders
```
![image](https://github.com/hindhujanaki/EXNO-3-DS/assets/148514666/74f5097f-261d-43da-a5f0-8379cb296e66)

```
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df
```
![image](https://github.com/hindhujanaki/EXNO-3-DS/assets/148514666/1adec4b1-0140-4a69-817e-38911c015236)

```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb
```
![image](https://github.com/hindhujanaki/EXNO-3-DS/assets/148514666/33379bcd-8c5b-40c3-af43-da92434a8cb7)

# Target Encoder
```
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new=te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc
```
![image](https://github.com/hindhujanaki/EXNO-3-DS/assets/148514666/092a6b84-ee98-4965-a2cb-a4eb08400b33)

# Data Transformation
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv('/content/Data_to_Transform.csv')
df
```
![image](https://github.com/hindhujanaki/EXNO-3-DS/assets/148514666/b64c3557-fa2f-4abe-8c4c-4d647f7b183d)

```
df.skew()
```
![image](https://github.com/hindhujanaki/EXNO-3-DS/assets/148514666/936c7662-0a08-4616-9ffc-a7f64458e465)

```
np.reciprocal(df["Moderate Positive Skew"])

```
![image](https://github.com/hindhujanaki/EXNO-3-DS/assets/148514666/f9646c39-c3a5-4b89-916e-ba700fcdb0f0)

```
np.sqrt(df["Highly Positive Skew"])
```
![image](https://github.com/hindhujanaki/EXNO-3-DS/assets/148514666/c2ea0ff8-93fb-49aa-9dd2-6d1a88d8b8e8)

```
np.square(df["Highly Positive Skew"])
```
![image](https://github.com/hindhujanaki/EXNO-3-DS/assets/148514666/c7db82c6-a73f-46d1-929a-ea17dfba7711)

```
df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/hindhujanaki/EXNO-3-DS/assets/148514666/f736d027-85f8-4cf9-b664-d95072c58b7d)

```
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
df.skew()
```
![image](https://github.com/hindhujanaki/EXNO-3-DS/assets/148514666/43f713bc-7174-4e39-9cf6-6261ede858f2)

```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
![image](https://github.com/hindhujanaki/EXNO-3-DS/assets/148514666/60f43b93-049e-4da3-8efa-3d0702390f62)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
![image](https://github.com/hindhujanaki/EXNO-3-DS/assets/148514666/5194efef-e470-4c11-a4da-6837e4723155)

```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/hindhujanaki/EXNO-3-DS/assets/148514666/e2172964-d31e-40b6-8fb7-10304beffca2)

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![image](https://github.com/hindhujanaki/EXNO-3-DS/assets/148514666/0f8a4fcf-d3da-4872-9e31-4ce61c7a4451)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/hindhujanaki/EXNO-3-DS/assets/148514666/b6b9ec07-6bbb-49cc-b563-805e7f4207dc)

```

df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/hindhujanaki/EXNO-3-DS/assets/148514666/2a3f2b6e-cc13-45bd-97ab-f7eb3735dd38)

```
sm.qqplot(df['Highly Negative Skew_1'],line='45')
plt.show()
```
![image](https://github.com/hindhujanaki/EXNO-3-DS/assets/148514666/da01cbad-0880-4b08-b452-90490484b6d3)

```

dt=pd.read_csv("/content/titanic_dataset.csv")
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45')
plt.show()
```
![image](https://github.com/hindhujanaki/EXNO-3-DS/assets/148514666/f64bdabd-db9d-46e3-b288-830123e8ed23)

```
sm.qqplot(dt['Age_1'],line='45')
plt.show()
```
![image](https://github.com/hindhujanaki/EXNO-3-DS/assets/148514666/3516d64a-1b72-4592-b628-d94f53408374)


# RESULT:
Thus perform Feature Encoding and Transformation process is executed successfully.

       
