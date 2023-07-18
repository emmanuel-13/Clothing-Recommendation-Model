import streamlit as st
import pickle as pk
from streamlit_option_menu import option_menu
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



cloth_model = pk.load(open('clothing.txt', 'rb'))
le = LabelEncoder()
st_x = StandardScaler()

dat = pd.read_csv('Clothing.csv')
new_data = dat.drop(columns=['Unnamed: 0', 'Clothing ID', 'Positive Feedback Count', 'Division Name', 'Review Text', 'Title'])
new_data.dropna(inplace=True)
a = new_data['Department Name']
b = new_data['Class Name']
c = new_data['Age']
d = new_data['Rating']
e = new_data['Recommended IND']
a_encoded = le.fit_transform(a)
b_encoded = le.fit_transform(b)

mydataset = {
    "Age": c,
    'Rating': d,
    'Department Name': a_encoded,
    'Class Name': b_encoded,
    'Recommended IND': e
 }
df = pd.DataFrame(mydataset)

x = df.iloc[:, [0,1,2,3]].values
y = df.iloc[:, 4].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
x_train = st_x.fit_transform(x_train)

department_name_mapping = {'Bottoms': 0, 'Dresses': 1, 'Intimate': 2, 'Jackets': 3, 'Tops': 4, 'Trend': 5}
class_name_mapping = {'Blouses': 0, 'Casual bottoms': 1, 'Chemises': 2, 'Dresses': 3, 'Fine gauge': 4, 'Intimates': 5,
                      'Jackets': 6, 'Jeans': 7, 'Knits': 8, 'Layering': 9, 'Legwear': 10, 'Lounge': 11, 'Outerwear': 12,
                      'Pants': 13, 'Shorts': 14, 'Skirts': 15, 'Sleep': 16, 'Sweaters': 17, 'Swim': 18, 'Trend': 19}


with st.sidebar:
    option = option_menu('Menu', ['Home', 'Model for Predicting', 'Add to cart', 'Payment'],
                         icons=['robot', 'robot', 'cart', 'cash'])

if option == 'Home':
    img1 = Image.open('pic.jpg')
    img2 = Image.open('image1.jpg')
    img3 = Image.open('img3.jpg')
    img4 = Image.open('img2.jpg')

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.image(img1)

    with col2:
        st.image(img2)

    with col3:
        st.image(img3)

    with col4:
        st.image(img4)


    st.title("WOMEN'S CLOTHING PREDICTION SITE")
    st.subheader("""
    Welcome to Keyla's women clothing prediction website...
    This site gives a detailed predicting for different dresses and their recommendations for different age groups.
    """)
    st.write('Before you proceed, click on model prediction to determine if our products are highly recommended for you')


if option == 'Model for Predicting':
    img1 = Image.open('img4.jpg')
    img2 = Image.open('img6.jpg')
    img3 = Image.open('Clothes.jpg')
    col5, col6, col7 = st.columns(3)
    with col5:
        st.image(img1)

    with col6:
        st.image(img2)

    with col7:
        st.image(img3)


    st.subheader('This model is designed to make your online shopping very easier')
    col1, col2, col3, col4 = st.columns(4)


    with col1:
        Age = st.text_input('Enter your age')

    with col2:
        Rating = st.selectbox(options=[1,2,3,4,5], label='choose the rating')

    with col3:
        DepartmentName = st.selectbox(options=['Bottoms', 'Dresses', 'Intimate', 'Jackets', 'Tops', 'Trend'],
                                        label='select the department name')

    with col4:
        ClassName = st.selectbox(options=['Blouses', 'Casual bottoms', 'Chemises', 'Dresses', 'Fine gauge', 'Intimates',
                                            'Jackets', 'Jeans', 'Knits', 'Layering', 'Legwear', 'Lounge', 'Outerwear',
                                            'Pants', 'Shorts', 'Skirts', 'Sleep', 'Sweaters', 'Swim', 'Trend'],
                                   label='select the class type')
    women_clothing = ''
    if st.button('Predict'):
       department_name_encoded = department_name_mapping[DepartmentName]
       class_name_encoded = class_name_mapping[ClassName]
       cloth_prediction = cloth_model.predict(st_x.transform([[Age, Rating, department_name_encoded, class_name_encoded]]))

       if (cloth_prediction[0] ==1):
           women_clothing = 'The cloth is recommended, You can go ahead to add it to your cart'
           st.success(women_clothing)

       else:
           women_clothing = 'The cloth is not recommended, You can select another class of dress.'
           st.warning(women_clothing)


if option == 'Add to cart':
    img6 = Image.open('img5.jpg')
    st.image(img6)
    st.write('Browse our categories and discover best our our products.')
    if st.button('Start Shopping'):
        img10 = Image.open('image1.jpg')
        img11 = Image.open('img3.jpg')
        img12 = Image.open('img2.jpg')
        img13 = Image.open('img4.jpg')
        img14 = Image.open('img6.jpg')
        img15 = Image.open('Clothes.jpg')
        col1, col2, col3, col4, col5, col6 = st.columns(6)

        with col1:
            st.image(img10, caption='N5000')

        with col2:
            st.image(img11, caption='N10,000')

        with col3:
            st.image(img12, caption='N15,000')

        with col4:
            st.image(img13, caption='N8,000')

        with col5:
            st.image(img14, caption='N3,500')

        with col6:
            st.image(img15, caption='N2000')




if option == 'Payment':
    st.write('Click here to make payments')
    st.button('Make Payments')




st.markdown('''
<style>
.css-h5rgaw.egzxvld1{
visibility: hidden;
}
</style>
''', unsafe_allow_html=True)
