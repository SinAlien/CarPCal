import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OrdinalEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

# تعریف ستون‌های مورد استفاده
columns = ['Brand', 'Model_Year', 'Mileage', 'Color', 'Body_Condition']

# خواندن داده‌ها
df = pd.read_csv('car_fin.csv')

# انتخاب ویژگی‌های مورد نیاز
df = df[columns + ['Price']]

# حذف مقادیر Null
df.dropna(inplace=True)

# محاسبه میانگین قیمت هر برند
brand_mean_price = df.groupby('Brand')['Price'].mean().to_dict()

# تابع تبدیل برند به میانگین قیمت
def brand_encoder(X):
    return np.array([brand_mean_price.get(b, np.mean(list(brand_mean_price.values()))) for b in X.iloc[:, 0]]).reshape(-1, 1)

# تابع محاسبه سن خودرو
def year(x):
    return 1404 - x if isinstance(x, (int, float)) else 1404 - x.astype(int)

# دسته‌بندی‌های ترتیبی برای `Body_Condition`
body_condition_categories = [
    "دوررنگ", "رنگ‌شدگی", "صافکاری بی‌رنگ", "خط و خش جزیی", "سالم و بی‌خط و خش"
]

# تعریف `ColumnTransformer` برای پردازش ویژگی‌ها
preprocessor = ColumnTransformer(transformers=[
    ('brand', FunctionTransformer(brand_encoder, validate=False), ['Brand']),  # تبدیل برند به عدد
    ('years', FunctionTransformer(year, validate=False), ['Model_Year']),  # محاسبه سن خودرو
    ('mileage', StandardScaler(), ['Mileage']),  # استانداردسازی کارکرد خودرو
    ('color', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ['Color']),  # کدگذاری رنگ
    ('body_condition', OrdinalEncoder(categories=[body_condition_categories]), ['Body_Condition'])  # کدگذاری شرایط بدنه
])

# ایجاد `Pipeline`
pipeline = Pipeline([
    ('preprocessor', preprocessor),  # پردازش ویژگی‌ها
    ('model', LinearRegression())    # مدل رگرسیون خطی
])

# تفکیک داده‌ها
X = df[columns]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# آموزش مدل
pipeline.fit(X_train, y_train)

# ذخیره مدل
joblib.dump(pipeline, 'car_price_pipeline.pkl')

print("✅ Pipeline ذخیره شد!")

# تعریف کلاس PrepProcesor برای استفاده در برنامه اصلی
class PrepProcesor:
    def __init__(self):
        self.pipeline = joblib.load('car_price_pipeline.pkl')

    def transform(self, data):
        # تبدیل داده‌های ورودی به فرمت مناسب برای پیش‌بینی
        return self.pipeline.named_steps['preprocessor'].transform(data)

    def predict(self, data):
        # پیش‌بینی قیمت
        return self.pipeline.predict(data)