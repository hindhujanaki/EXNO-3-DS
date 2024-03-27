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
![image](https://github.com/hindhujanaki/EXNO-3-DS/assets/148514666/09ba6aec-36f9-4159-a7b8-8b0ffc868052)
```
```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
e1.fit_transform(df[["ord_2"]])
pm=["Hot","Warm","Cold"]
e1=OrdinalEncoder(categories=[pm])
![image](https://github.com/hindhujanaki/EXNO-3-DS/assets/148514666/cc8e1022-0a9c-4639-9ccb-563b99c7fbb4)
```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
![image](https://github.com/hindhujanaki/EXNO-3-DS/assets/148514666/0230b89d-948c-4596-9532-b86a03609466)
```
```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(df[['ord_2']])
dfc
![image](https://github.com/hindhujanaki/EXNO-3-DS/assets/148514666/2fbe8bd9-9687-4daa-9b43-fa9cf490ea74)
```
```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
![image](https://github.com/hindhujanaki/EXNO-3-DS/assets/148514666/94b75cd8-f4d3-4fd6-a4a6-78ad41f8125d)

df2=pd.concat([df2,enc],axis=1)
df2
![image](https://github.com/hindhujanaki/EXNO-3-DS/assets/148514666/9d3935f8-b3c1-4b64-aaa0-5bff8e5c7845)

pd.get_dummies(df2,columns=["nom_0"])
![image](https://github.com/hindhujanaki/EXNO-3-DS/assets/148514666/fedeb68a-a590-4ef5-80da-7811b01fb637)

pip install --upgrade category_encoders
![image](https://github.com/hindhujanaki/EXNO-3-DS/assets/148514666/478de62d-5592-4c98-9d3b-c82cf07dc0bc)

from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df
![image](https://github.com/hindhujanaki/EXNO-3-DS/assets/148514666/be0ecc45-9aae-4679-95f1-1fb1b1840ac8)

df=pd.read_csv("/content/data.csv")
dfb=pd.concat([df,nd],axis=1)
dfb
![image](https://github.com/hindhujanaki/EXNO-3-DS/assets/148514666/70d193f1-cd68-4a04-ae7f-6c604dd84cb5)

from category_encoders import TargetEncoder
te=TargetEncoder()
new=te.fit_transform(X=cc["City"],y=cc["Target"])
cc
![image](https://github.com/hindhujanaki/EXNO-3-DS/assets/148514666/481c8f67-a506-48cf-b24c-d9a7eb040f7a)
## Data transformation
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform (1).csv")
df
![image](https://github.com/hindhujanaki/EXNO-3-DS/assets/148514666/6ccdb819-ed44-4866-871f-d7a5e9f2eac2)
df.skew()
![image](https://github.com/hindhujanaki/EXNO-3-DS/assets/148514666/6fe91bd0-4275-4c2d-98e1-3ca0ad0e123a)
np.log(df["Highly Positive Skew"])
![image](https://github.com/hindhujanaki/EXNO-3-DS/assets/148514666/f10f5b8a-8314-412d-9fc9-83188a8f9fe3)
np.reciprocal(df["Moderate Positive Skew"])
![image](https://github.com/hindhujanaki/EXNO-3-DS/assets/148514666/81509dde-d6c5-461d-90bf-ca9128d09811)
np.sqrt(df["Highly Positive Skew"])
![image](https://github.com/hindhujanaki/EXNO-3-DS/assets/148514666/ad6733b4-9c0e-49af-8e1e-e8a7c51c8a63)
np.square(df["Highly Positive Skew"])
![image](https://github.com/hindhujanaki/EXNO-3-DS/assets/148514666/7e6bfc01-2a6a-482b-82f0-7f0fb78647a5)
df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
df
![image](https://github.com/hindhujanaki/EXNO-3-DS/assets/148514666/9eac4cde-6dd6-4bff-824a-5ddb5392ee8f)
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
df.skew()
![image](https://github.com/hindhujanaki/EXNO-3-DS/assets/148514666/53a35206-82e4-4b0f-82cd-947b9851a9d2)
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
![image](https://github.com/hindhujanaki/EXNO-3-DS/assets/148514666/11645e21-5b69-44ed-a9f0-55c3f70ca9cb)
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line="45")
plt.show()
![image](https://github.com/hindhujanaki/EXNO-3-DS/assets/148514666/5d152d4b-8831-40d6-84c6-4dbdf46f8e4c)
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line="45")
plt.show()
![image](https://github.com/hindhujanaki/EXNO-3-DS/assets/148514666/37f773ee-7839-44bc-a86e-fbfbedc406c7)
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution="normal",n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line="45")
plt.show()
![image](https://github.com/hindhujanaki/EXNO-3-DS/assets/148514666/e2c18d76-ba04-4c30-b368-71937f9ab2bb)
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
![image](https://github.com/hindhujanaki/EXNO-3-DS/assets/148514666/a3587413-5c0a-497a-bb67-306a27f25e96)
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
![image](https://github.com/hindhujanaki/EXNO-3-DS/assets/148514666/0d29b565-e51b-4ec9-ba73-a02a66e42ea0)
sm.qqplot(df['Highly Negative Skew_1'],line='45')
plt.show()
![image](https://github.com/hindhujanaki/EXNO-3-DS/assets/148514666/cadb76d0-8819-4171-9f86-0b160f6a9626)
dt=pd.read_csv("/content/titanic_dataset.csv")
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45')
plt.show()
![image](https://github.com/hindhujanaki/EXNO-3-DS/assets/148514666/289c721b-8624-4ef5-b683-85e681876c7b)
sm.qqplot(dt['Age_1'],line='45')
plt.show()
![image](https://github.com/hindhujanaki/EXNO-3-DS/assets/148514666/f4a7873e-3e30-44d9-bb5d-59dc625d5e9e)
```
# RESULT:
       Thus perform Feature Encoding and Transformation process is executed successfully.



       
