import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from with_pipe import columns,PrepProcesor

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
#model = joblib.load('car_price_pipeline.pkl')
preprocessor = joblib.load('car_price_pipeline.pkl')
#preprocessor = PrepProcesor()

# تنظیم صفحه Streamlit
st.title("تحلیل داده‌های خودرو و پیش‌بینی قیمت")

Brand = st.selectbox(
    "Brand?",
    (
    'پژو 206 SD V8',
    'پژو 206 تیپ ۲',
    'پژو 206 تیپ ۳',
    'پژو 206 تیپ ۵',
    'پژو 207i دنده\u200cای',
    'پژو 207i پانوراما دنده\u200cای',
    'پژو 405 GLX - دوگانه سوز CNG',
    'پژو 405 GLX بنزینی',
    'پژو 405 جی ال ایکس (GLX)',
    'پژو 405 SLX موتورTU5',
    'پژو پارس LX TU5',
    'پژو پارس ساده',
    'پژو پارس سال',
    'پژو پارس دوگانه سوز',
    'پژو پارس موتور جدید XU7P',
    'پژو پارس موتور جدید XU7P (سفارشی)',
    'پراید 111 SE',
    'پراید 131 EX',
    'پراید 131 SE',
    'پراید 131 دوگانه سوز',
    'پیکان وانت CNG',
    'تندر 90 (L90 لوگان) E2',
    'تیبا 2 (هاچبک) EX',
    'تیبا صندوق\u200cدار SX',
    'دنا پلاس اتوماتیک',
    'رانا LX',
    'رانا پلاس',
    'زامیاد Z 24 دوگانه سوز',
    'ساینا دنده\u200cای EX',
    'ساینا دنده\u200cای S',
    'سمند LX EF7',
    'سمند LX EF7 بنزینی',
    'سمند LX ساده',
    'سمند سورن پلاس',
    'سیتروئن زانتیا 2000cc',
    'شاهین G',
    'کوییک GX',
    'کوییک RS',
    'کوییک دنده\u200cای R',
    'کوییک دنده\u200cای S',
    'کوییک دنده\u200cای ساده',
    'وانت پراید 151 SE',
    'وانت نیسان'
    ),
)
st.write("You selected:", Brand)

Color = st.selectbox(
    "color?",
    ('خاکستری',
 'سفید',
 'سفید صدفی',
 'نقره\u200cای',
 'نوک\u200cمدادی',
 'آبی',
 'مشکی',
 'دلفینی',
 'زرد',
 'بژ'),
)
st.write("You selected:", Color)



Model_Year = st.select_slider(
    "Select Model Year: ",
    options=[
        1403,
        1402,
        1401,
        1400,
        1399,
        1398,
        1397,
        1396,
        1395,
        1394,
        1393,
        1392,
        1391,
        1390,
        1389,
        1388,
        1387,
        1386,
        1385,
        1384,
        1383,
        1382,
        1381,
        1380
        ],
    help="Select the production year of the car.",  # پیام راهنما
    label_visibility="visible",  # نمایش لیبل
)
st.write("Model is: ", Model_Year)

Body_Condition = st.radio(
    "How is Body Condition?",
    ['سالم و بی\u200cخط و خش',
    'خط و خش جزیی',
    'صافکاری بی\u200cرنگ',
    'رنگ\u200cشدگی',
    'دوررنگ',
    ],  
        index=0,
)

st.write("You selected:", Body_Condition)

Mileage = st.number_input(
    "Insert KM usage of the car", value=None, placeholder="Type a number..."
)
st.write("The current KM usage is ", Mileage)


def predict():
    # بررسی مقدار کارکرد خودرو
    if Mileage is None:
        st.error("لطفاً کارکرد خودرو را وارد کنید.")
        return

    # ایجاد یک DataFrame از ورودی‌های کاربر
    data = [[Brand, Model_Year, Mileage, Color, Body_Condition]]  # تغییر داده شد
    X = pd.DataFrame(data, columns=columns)

    
    # نمایش داده‌های ورودی برای دیباگ
    st.write("داده‌های ورودی:", X)
    X_transformed = preprocessor.named_steps['preprocessor'].transform(X)
    st.write("داده‌های پردازش‌شده:", X_transformed)
    # پیش‌بینی قیمت
    try:
        prediction = preprocessor.predict(X)
        st.success(f"قیمت پیش‌بینی‌شده: {prediction[0]:,.0f} تومان")
    except Exception as e:
        st.error(f"خطا در پیش‌بینی: {e}")

# دکمه پیش‌بینی
if st.button('پیش‌بینی قیمت'):
    predict()