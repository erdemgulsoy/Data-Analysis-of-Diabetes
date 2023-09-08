# author : Mustafa Erdem Gülsoy
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = pd.read_csv("datasets/diabetes.csv")


# DataFrame için ilk olarak kategorik ve numerik değişkenleri ayırıyoruz ;
def grab_col_names(dataframe, cat_th=5, car_th=20) :
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.

    Parameters
    ----------
    dataframe : dataframe
        değişken isimleri alınmak istenen dataframe'dir.
    cat_th : int, float
        numerik fakat kategorik olan değişkenlerdir.
    car_th : int, float
        kategorik fakat kardinal olan değişkenler için sınıf eşik değeri.

    Returns
    -------
        cat_cols : list
            Kategorik değişken listesi.
        num_cols : list
            Numerik değişkeen listesi.
        cat_but_car : list
            Kategorik görünümlü kardinal değişken listesi.

    Notes
    ------
        cat_cols + num_cols +cat_but_car = toplam değişken sayısı.
        num_but_cat, cat_cols'un içerisinde.

    """

    cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]

    num_but_cat = [col for col in df.columns if df[col].dtypes in ["int64", "float64", "uint8"] and df[col].nunique() < cat_th]

    cat_but_car = [col for col in df.columns if str(df[col].dtypes) in ["category", "object"] and df[col].nunique() > car_th]

    cat_cols += num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in df.columns if col not in cat_cols and df[col].dtypes in ["int64", "float64", "uint8"]]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# DataFrame için ön bilgi alma işlemi ;
def check_df (dataframe, head=5) :
    print("#################### Shape ##################")
    print(dataframe.shape)
    print("#################### Type ##################")
    print(dataframe.dtypes)
    print("#################### Head ##################")
    print(dataframe.head(head))
    print("#################### Tail ##################")
    print(dataframe.tail(head))
    print("#################### NA ##################")
    print(dataframe.isnull().sum())
    print("#################### Quantiles ##################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
check_df(df)

def cat_summary2(dataframe, col_name, plot=False) :
    print(pd.DataFrame({col_name : dataframe[col_name].value_counts(),
                        "Ratio"  : 100 * dataframe[col_name].value_counts() / len(dataframe) }))
    print("###############################################")

    if plot :
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)
for col in cat_cols :
    cat_summary2(df, col, True)

def num_summary2(dataframe, numerical_col, plot=False) :
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot :
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)
for col in num_cols :
    num_summary2(df, col, True)

# Kategorik değişkenlere göre hedef değişkenin ortalaması ;
for col in cat_cols :
    print(df.groupby(col).agg({"Outcome": "mean"}))

# Hedef değişkene göre numerik değişkenlerin ortalaması ;
for col in num_cols :
    print(df.groupby("Outcome").agg({col: "mean"}))

#################################################
#################################################

# Outliers Analizi ;
# Aykırı değer kontrolü için up ve down oranlarını belirliyoruz.
def outlier_thresholds(dataframe, col_name, q1=0.10, q3=0.90) :

    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)

    iqr = quartile3 - quartile1 # InterQuartileRange

    up = quartile3 + iqr * 1.5  # formüle göre üst sınır.
    down = quartile1 - iqr * 1.5  # formüke göre alt sınır.

    return up, down
# Değişkenlerin aykırı değerleri varsa True, yoksa False veriyor. (check outlier)
def check_outlier(df, col) :
    up, down = outlier_thresholds(df, col)
    print(col, " : ", df[(df[col] < down) | (df[col] > up)].any(axis=None))
for col in df.columns:
    check_outlier(df, col)



# Outliers re-assignment with thresholds (Baskılama Yöntemi) ;
def replace_with_thresholds(dataframe, col_name) :
    up, down = outlier_thresholds(df, col)
    dataframe.loc[(dataframe[col_name] < down), col_name] = down
    dataframe.loc[(dataframe[col_name] > up), col_name] = up
for col in df.columns :
    replace_with_thresholds(df, col)
# Aykırı değerlerden kurtulduk.

#################################################
#################################################

# Missing Value Analizi ;

# Eksik değer var mı ?
def missing_values_table(dataframe, na_name = False) :
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=["n_miss", "ratio"])
    print(missing_df, end="\n")

    if na_name :
        return na_columns
missing_values_table(df)
# Eksik Değer Yok.


# Korelasyon Analizi ; (Eksik Değer Olmadığı için Bir Bilgi İçermeyecek.)
msno.bar(df)
plt.show(block=True)

msno.matrix(df)
plt.show(block=True)

msno.heatmap(df)
plt.show(block=True)

# Veri setinde eksik gözlem bulunmamakta ama Glikoz, Insulin vb. değişkenlerde 0 değeri içeren gözlem birimleri eksik değeri ifade ediyor olabilir.
# Örneğin; bir kişinin glikoz veya insulin değeri 0
# olamayacaktır. Bu durumu dikkate alarak sıfır değerlerini ilgili değerlerde NaN olarak atama yapıp sonrasında eksik
# değerlere işlemleri uygulayabiliriz.

cols_to_check_for_nan = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[cols_to_check_for_nan] = df[cols_to_check_for_nan].replace(0, np.nan)
# Yukarıda 0 olmaması gereken değerlerin sıfır olduğu yerleri NaN ile değiştirdik ve tekrardan eksik değer kontrolü yapacağız ;

# Eksik değer var mı ?
def missing_values_table(dataframe, na_name = False) :
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=["n_miss", "ratio"])
    print(missing_df, end="\n")

    if na_name :
        return na_columns
missing_values_table(df)
# Birçok Eksik Değer var. Bunları analiz edip gerekli işlemleri yapacağız.


# Korelasyonu yüksek olanların önemi daha azdır. Buna bakalım ;
msno.heatmap(df)
plt.show(block=True)

# Eksik Değerlerin Bağımlı Değişken Analizi;
na_cols = missing_values_table(df, True)

def missing_vs_target(dataframe, target, na_columns) :
    temp_df = dataframe.copy()

    for col in na_columns :
        temp_df[col + "_NA_FLAG"] = np.where(temp_df[col].isnull(), 1, 0)

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags :
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(), "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")
missing_vs_target(df, "Outcome", na_cols)
# çok da göze çarpan bir değişken yok.


# Standart Scaler uygulayalım ;
scaler = StandardScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)


# KNN ile Eksik değerleri doldurmayı deneyeceğiz;
imputer = KNNImputer(n_neighbors=5)
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)


# Eski Haline Getirelim ;
df = pd.DataFrame(scaler.inverse_transform(df), columns=df.columns)

# Tüm eksik Değerleri en yakın 5 komşusuna göre doldurmuş bulunuyoruz.

#################################################
#################################################

# Değişken Türetelim ;
df.loc[(df["BMI"] > 10) & (df["BMI"] <= 20), "NEW_BMI_CAT"] = "underweight"
df.loc[(df["BMI"] > 20) & (df["BMI"] <= 25), "NEW_BMI_CAT"] = "healthly"
df.loc[(df["BMI"] > 25) & (df["BMI"] <= 30), "NEW_BMI_CAT"] = "overweight"
df.loc[(df["BMI"] > 30) & (df["BMI"] <= 40), "NEW_BMI_CAT"] = "obese"
df.loc[(df["BMI"] > 40) & (df["BMI"] <= 60), "NEW_BMI_CAT"] = "very_obese"

df.loc[(df["BloodPressure"] < 60), "NEW_PRESS_CAT"] = "low"
df.loc[(df["BloodPressure"] >= 60) & (df["BloodPressure"] < 80), "NEW_PRESS_CAT"] = "optimal"
df.loc[(df["BloodPressure"] >= 80) & (df["BloodPressure"] < 90), "NEW_PRESS_CAT"] = "normal"
df.loc[(df["BloodPressure"] >= 90) & (df["BloodPressure"] < 110), "NEW_PRESS_CAT"] = "high"
df.loc[(df["BloodPressure"] >= 110), "NEW_PRESS_CAT"] = "very_high"

df.head()

# Yeni iki değişken türettik. şimdi de bu yeni değişkenleri kolonları güncellemeliyiz ;
cat_cols, num_cols, cat_but_car = grab_col_names(df)

#################################################
#################################################

# Encoding işlemlerini gerçekleştirelim;

# Label Encoding  ;
binary_cols = [col for col in df.columns if df[col].dtype not in ["int64", "float64"] and df[col].nunique() == 2 ]
# binary column olmadığı için bu adımı atlıyoruz.


# Rare Encoding ;
def rare_analyser(dataframe, target, cat_cols) :
    for col in cat_cols :
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"Count": dataframe[col].value_counts(),
                            "Ratio": 100 * (dataframe[col].value_counts() / len(dataframe)),
                            "Target_Mean": dataframe.groupby(col)[target].mean()}), end="\n\n\n")
def rare_encoder(dataframe, rare_perc) :
    temp_df = dataframe.copy()
    # burada ilk önce kategorik ve içinde 0.01'in altında oran olanlar varsa bu KOLONLARI rare_columns olarak alıyoruz.
    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == "O"
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    # içinde 0.01'den az oran olan kategoriler olduğunu bildiğimiz kolonların içinde gezerek bu kategorileri seçiyoruz ve bunları rare_labels içine atıyoruz. unutmayalım ; rare_columsn = içinde 0.01'den az oranda kategori olduğunu bildiğimiz kolonlar, rare_labels = bu kolonlardaki 0.01 oranındaki kategoriler.
    for var in rare_columns :
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), "Rare", temp_df[var])


    return temp_df
rare_analyser(df, "Outcome",cat_cols)
df = rare_encoder(df, 0.01)


# One-Hot Encoder ;

# Kategori sayısı 2 ile 10 arasında olanları ohe_cols olarak toplayalım ve işlem uygulayalım ;
ohe_cols = [col for col in df.columns if df[col].nunique() > 2 and df[col].nunique() <= 10]
def one_hot_encoder(dataframe, categorial_cols, drop_first=True) :
    dataframe = pd.get_dummies(dataframe, columns=categorial_cols, drop_first=drop_first)
    return dataframe
df = one_hot_encoder(df, ohe_cols)

# Kategorileri güncelliyoruz.
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Yeni değişkenler geldiği için tekrardan rare analyser yapmalıyız ;
rare_analyser(df,"Outcome", cat_cols)

# Gereksiz, kullanışsız değişken var mı diye baktık. Eğer varsa ya rare yapacaktık ya da silecektik. Ama burada yok.
useless_cols = [col for col in df.columns if df[col].nunique() == 2 and (df[col].value_counts() / len(df) < 0.01).any(axis=None) ]

#################################################
#################################################

# Numerik değişkenleri Standartlaştırıyoruz ;
scaler = StandardScaler()
df[num_cols]  = scaler.fit_transform(df[num_cols])
df[num_cols].head()

#################################################
#################################################

# Model Oluşturuyoruz ;

y = df["Outcome"]
X = df.drop("Outcome", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)
# accuracy score : 0.835


# Hiçbir şey yapmasaydık accuracy score kaç çıkacaktı ?

df2 = pd.read_csv("datasets/diabetes.csv")
y2 = df2["Outcome"]
X2 = df2.drop("Outcome", axis=1)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.30)
rf_model2 = RandomForestClassifier(random_state=46).fit(X_train2, y_train2)
y_pred2 = rf_model2.predict(X_test2)
accuracy_score(y_pred2, y_test2)
# accuracy score : 0.740


# Yeni türettiğimiz değişkenlerin önem ve işe yarama oranını grafikle inceleyelim ;
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig('importances.png')
plot_importance(rf_model, X_train)

