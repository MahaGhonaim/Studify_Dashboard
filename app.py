import streamlit as st
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.dates import DateFormatter

# =============================================
# إعدادات الواجهة وتخصيص المظهر الداكن
# =============================================

# تخصيص المظهر العام للواجهة الداكنة
st.markdown(
    """
    <style>
        /* الخلفية الرئيسية */
        .main {
            background-color: #0E1117;
            color: white;
        }
        
        /* تخصيص الألوان للعناصر */
        .stMetric {
            background-color: #1E1E1E !important;
            border-radius: 10px;
            padding: 15px;
        }
        
        .stMetric label {
            color: white !important;
        }
        
        .stMetric value {
            color: white !important;
        }
        
        /* تخصيص الرسوم البيانية */
        .stPlot {
            background-color: transparent !important;
        }
        
        /* تخصيص العناوين */
        h1, h2, h3, h4, h5, h6 {
            color: white !important;
        }
        
        /* تخصيص النصوص */
        p, div {
            color: white !important;
        }
        
        /* الحفاظ على السايدبار كما هو */
        .css-1v3fvcr {
            background-color: #0E1117;
        }
        
        /* تخصيص الأزرار */
        .stButton button {
            background-color: #3838ab;
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# =============================================
# تحميل البيانات
# =============================================

@st.cache_data
def load_data():
    excel_file = "/content/UdemyDWH.xlsx"
    df_courses = pd.read_excel(excel_file, sheet_name='DimCourses')
    df_enrollments = pd.read_excel(excel_file, sheet_name='FactEnrollment')
    df_users = pd.read_excel(excel_file, sheet_name='DimUsers')
    df_dates = pd.read_excel(excel_file, sheet_name='DimDate')
    return df_courses, df_enrollments, df_users, df_dates

df_courses, df_enrollments, df_users, df_dates = load_data()

# =============================================
# رأس الصفحة والشريط الجانبي
# =============================================

# عنوان أزرق، بولد، ومحاذي للشمال مع اللوجو بجانب العنوان مباشرة
title_html = """
<h2 style='text-align: left; margin-top: 0; margin-bottom: 0px; color: #3838ab ; font-weight: bold; display: inline-block;'>
    📈 Studify Dashboard
</h2>
"""
st.markdown(title_html, unsafe_allow_html=True)
st.markdown("""
This dashboard, developed by ITI Mansoura students as part of their Power BI Developer training
""")

# شريط جانبي مع صورة اللوجو
logo_html = """
<div style="text-align:left;">
    <img src="data:image/png;base64,{}" style="width: 200px; height: 200px">
</div>
"""

def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

encoded_image = get_image_base64("/content/studify00.png")
st.sidebar.markdown(logo_html.format(encoded_image), unsafe_allow_html=True)
st.sidebar.header("🎛️ Filters")

# =============================================
# قسم المقاييس الرئيسية
# =============================================

# معالجة البيانات للمقاييس
Total_Courses = df_courses['CourseId_BK'].count()
Total_Students = df_users[df_users['IsStudent'] == 1].count()['IsStudent']
Total_Instructor = df_users[df_users['IsInstructor'] == 1].count()['IsInstructor']
df_courses['CurrentPrice'] = pd.to_numeric(df_courses['CurrentPrice'], errors='coerce')
MaxCourse_price = df_courses['CurrentPrice'].max()
AVGCourse_price = df_courses['CurrentPrice'].mean()
Student_AVG_Age = int(df_users[df_users['IsStudent'] == 1]['Age'].mean())
Student_countries = len(df_users[df_users['IsStudent'] == 1]['CountryName'].unique())
df_courses['Rating'] = pd.to_numeric(df_courses['Rating'], errors='coerce')
Avg_courses_Rating = df_courses['Rating'].mean()

# عرض العنوان والوصف
st.subheader(" 📊 Metrics")
st.markdown("""
This section displays key metrics related to courses and students on the platform. 
""")

# تنسيق القيم
formatted_cours = "{:.1f}K".format(Total_Courses / 1000)
formatted_st = "{:.1f}K".format(Total_Students / 1000)
formatted_price = "{:.1f}$".format(AVGCourse_price)
formatted_max = "{:.1f}$".format(MaxCourse_price)
formatted_rating = "{:.1f}".format(Avg_courses_Rating)

# عرض المقاييس في أعمدة
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Courses", value=formatted_cours)
col2.metric("Avg courses Rating", value=formatted_rating)
col3.metric("Max Course price", value=formatted_max)
col4.metric("AVG Course price", value=formatted_price)

col5, col6, col7, col8 = st.columns(4)
col5.metric("Total Students", value=formatted_st)
col6.metric("Total Instructor", value=Total_Instructor)
col7.metric("Student AVG Age", value=Student_AVG_Age)
col8.metric("NO. Students countries", value=Student_countries)

# =============================================
# الرسم البياني 1: أفضل الكورسات حسب المشتركين
# =============================================

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import streamlit as st

# Sidebar filters
course_level = st.sidebar.selectbox(
    'Select Course Level',
    options=df_courses.loc[df_courses['CourseLevel'].notna() & (df_courses['CourseLevel'] != 0), 'CourseLevel'].unique()
)

category = st.sidebar.selectbox(
    'Select Category',
    options=df_courses.loc[df_courses['Category'].notna() & (df_courses['Category'] != 0), 'Category'].unique()
)

# Filter data
filtered_courses = df_courses[
    (df_courses['CourseLevel'] == course_level) & 
    (df_courses['Category'] == category)
]

# Calculate total subscribers per course
top_courses = filtered_courses.groupby('Title')['NoSubscribers'].sum().sort_values(ascending=False).head(10)

# Shorten long titles
short_titles = [title[:25] + "..." if len(title) > 25 else title for title in top_courses.index]

# Color gradient based on number of subscribers
norm = mcolors.Normalize(vmin=top_courses.min(), vmax=top_courses.max())
cmap = cm.get_cmap('Blues')  # تدرج أزرق
colors = [cmap(norm(value)) for value in top_courses.values]

# Add description
st.subheader("📊 Top 10 Courses by Subscribers")
st.markdown(f"Courses with the highest number of subscribers in **{category}** category at **{course_level}** level.")

# Plot

plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(short_titles, top_courses.values, color=colors)

# تخصيص المظهر
ax.set_facecolor('#0E1117')
fig.patch.set_facecolor('#0E1117')
ax.spines['left'].set_color('white')
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')


# Clean up chart
ax.set_title('')
ax.spines[['top', 'right', 'left']].set_visible(False)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Show chart
st.pyplot(fig, use_container_width=True)


# =============================================
# الرسم البياني 2: حالة التسجيلات
# =============================================

# حساب توزيع حالة التسجيلات
status_counts = df_enrollments['Status'].value_counts()
labels = status_counts.index
sizes = status_counts.values

# ألوان مخصصة
colors = ['#3838ab', '#4a6fa5', '#5c86b0', '#6e9dbb']

# إنشاء الرسم البياني
fig, ax = plt.subplots(figsize=(7, 6))
wedges, _ = ax.pie(sizes, startangle=140, colors=colors)

# تخصيص المظهر
fig.patch.set_facecolor('#0E1117')
ax.set_facecolor('#0E1117')

# إضافة وسيلة الإيضاح
legend_labels = [f'{label}: {value} ({(value/sum(sizes))*100:.1f}%)' for label, value in zip(labels, sizes)]
ax.legend(wedges, legend_labels, title="Status", loc="center left", bbox_to_anchor=(1, 0.5), 
          facecolor='#1E1E1E', labelcolor='white')

# عرض الرسم البياني
st.subheader("📊 Course Enrollment Status")
st.markdown("Distribution of course enrollment statuses (Completed, In Progress, etc.)")
st.pyplot(fig, use_container_width=True)

# =============================================
# الرسم البياني 3: متوسط التقييم حسب الفئة
# =============================================

# فلترة البيانات
df_valid = df_courses[df_courses['Rating'].notna() & df_courses['Category'].notna()]

# حساب متوسط التقييم
top_categories = df_valid.groupby('Category')['Rating'].mean().sort_values(ascending=False).head(10)

# إنشاء الرسم البياني
fig, ax = plt.subplots(figsize=(10,6))
sns.barplot(x=top_categories.values, y=top_categories.index, palette="Blues_r", ax=ax)

# تخصيص المظهر
ax.set_facecolor('#0E1117')
fig.patch.set_facecolor('#0E1117')
ax.spines[['top', 'right', 'left']].set_visible(False)
ax.spines['left'].set_color('white')
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')


# عرض الرسم البياني
st.subheader("⭐ Average Rating by Top 10 Categories")
st.markdown("Average course rating for the top 10 categories with highest rating")
st.pyplot(fig, use_container_width=True)

# =============================================
# الرسم البياني 4: توزيع الطلاب حسب الدولة والعمر
# =============================================
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import streamlit as st

# إعداد Age Group
df_users['AgeGroup'] = pd.cut(df_users['Age'], bins=[0, 18, 25, 35, 50, 100], 
                              labels=["<18", "18-25", "26-35", "36-50", "50+"])

# اختيار Age Group من الـ sidebar
age_group = st.sidebar.selectbox(
    "Select Age Group",
    options=df_users['AgeGroup'].dropna().unique()
)

# اختيار عدد الدول اللي تظهر
top_n = st.sidebar.slider("Number of Top Countries to Show", min_value=5, max_value=20, value=10)

# تصفية البيانات حسب Age Group و الطلاب فقط
filtered_users = df_users[
    (df_users['IsStudent'] == 1) & 
    (df_users['AgeGroup'] == age_group)
]

# حساب عدد الطلاب حسب الدولة
country_counts = filtered_users['CountryName'].value_counts().sort_values(ascending=False).head(top_n)
country_counts = country_counts.sort_values()  # علشان يترسم من الأقل للأعلى

# رسم العنوان والوصف
st.subheader("🌍 Student Country Distribution")
st.markdown(f"Top **{top_n}** countries with highest number of students in **{age_group}** age group")

# تدرج الألوان حسب عدد الطلاب
norm = mcolors.Normalize(vmin=country_counts.min(), vmax=country_counts.max())
cmap = cm.get_cmap('Blues')
colors = [cmap(norm(value)) for value in country_counts.values]

# إنشاء الرسم البياني
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(country_counts.index, country_counts.values, color=colors)

# تخصيص المظهر
ax.set_facecolor('#0E1117')
fig.patch.set_facecolor('#0E1117')
#ax.spines['bottom'].set_color('white')
#ax.spines['left'].set_color('white')
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# إضافة الأرقام على الأعمدة
for i, v in enumerate(country_counts.values):
    ax.text(v + 1, i, str(v), va='center', fontsize=8)

# عرض في Streamlit
st.pyplot(fig, use_container_width=True)
st.subheader("")


# =============================================
# الرسم البياني 5: أفضل التصنيفات الفرعية
# =============================================

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import streamlit as st


# فلتر لاختيار اللغة
selected_language = st.sidebar.selectbox(
    'Select Language',
    options=df_courses['Language'].dropna().unique()  # تأكد من إزالة القيم الناقصة
)

# تصفية البيانات حسب اللغة المختارة
filtered_courses = df_courses[df_courses['Language'] == selected_language]

# حساب عدد الكورسات في كل ساب كاتيجوري
top_subcategories = filtered_courses['SubCategory'].value_counts().head(10)

# تحديد تدرج الألوان بناءً على عدد الكورسات
norm = mcolors.Normalize(vmin=top_subcategories.min(), vmax=top_subcategories.max())
cmap = plt.get_cmap('Blues')  # يمكن تغيير هذا إلى تدرجات أخرى مثل 'plasma', 'cool', 'Blues'
colors = [cmap(norm(value)) for value in top_subcategories.values]

# إنشاء الرسم البياني
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(top_subcategories.index, top_subcategories.values, color=colors)

# إضافة العنوان والمحاور


# تخصيص شكل الرسم
plt.xticks(rotation=45, ha='right')
plt.tight_layout()


# تخصيص المظهر
ax.set_facecolor('#0E1117')
fig.patch.set_facecolor('#0E1117')
#ax.spines['bottom'].set_color('white')
#ax.spines['left'].set_color('white')
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')


# عرض الرسم البياني
st.subheader("📊 Top 10 Subcategories by Number of Courses")
st.markdown(f"Top subcategories in **{selected_language}** language by number of courses")

# إزالة المربع المحيط بالرسم البياني
for spine in ['top', 'right', 'left']:
    ax.spines[spine].set_visible(False)

# عرض الرسم البياني داخل Streamlit
st.pyplot(fig, use_container_width=True)


st.subheader("")
