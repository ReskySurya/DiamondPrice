import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

st.set_option('deprecation.showPyplotGlobalUse', False)

st.title('Selamat Datang di Prediksi Harga Diamond')

menu = ['Home', 'Prediksi', 'Graph']
selected_menu = st.sidebar.selectbox('Menu', menu)

data = pd.read_csv('diamonds.csv')

if selected_menu == 'Home':
    st.subheader('Home')
    st.image('thumbnail.png', use_column_width=True)

    st.subheader('Sample of the Dataset:')
    st.write(data.head(10))

    def plot_bar_chart(data):
        fig, ax = plt.subplots()
        if np.issubdtype(data.dtype, np.number):
            data.plot(kind='bar', ax=ax)
        else:
            data.value_counts().plot(kind='bar', ax=ax)
        st.pyplot(fig)

    excluded_columns = ['cut', 'color', 'clarity']
    features_exclude_excluded_columns = [feature for feature in data.columns if feature not in excluded_columns]

    selected_feature = st.selectbox('Choose the feature to plot', features_exclude_excluded_columns)

    filtered_data = data[selected_feature]

    # Display highest, median, and lowest values
    st.write(f"Highest value for {selected_feature}: {filtered_data.max()}")
    st.write(f"Median value for {selected_feature}: {filtered_data.median()}")
    st.write(f"Lowest value for {selected_feature}: {filtered_data.min()}")

    plot_bar_chart(filtered_data)

elif selected_menu == 'Prediksi':
    st.subheader('Selamat Datang di Prediksi')
    st.image('predict.png', use_column_width=True)

    selected_prediction_method = st.selectbox('Pilih Metode Prediksi:', ['Carat, Depth, Table', 'Lebar, Panjang, Tinggi'])

    if selected_prediction_method == 'Carat, Depth, Table':

        x = data[['carat', 'depth', 'table']]
        y = data['price']

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        model_regresi = LinearRegression()
        model_regresi.fit(X_train, y_train)

        carat_input = st.number_input('Masukan Nilai Carat')
        depth_input = st.number_input('Masukan Nilai Depth')
        table_input = st.number_input('Masukan Nilai Table')

        if st.button('Hitung Prediksi'):
            X_input = np.array([[carat_input, depth_input, table_input]])
            price_prediction = model_regresi.predict(X_input)

            price_prediction_integer = int(price_prediction[0])

            st.subheader('Prediksi Harga Diamond :')
            st.write(f'Harga Prediksi Diamond adalah ${price_prediction_integer}')

    elif selected_prediction_method == 'Lebar, Panjang, Tinggi':

        x = data[['x', 'y', 'z']]
        y = data['price']

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        model_regresi = LinearRegression()
        model_regresi.fit(X_train, y_train)

        lebar_input = st.number_input('Masukan Nilai Lebar')
        panjang_input = st.number_input('Masukan Nilai Panjang')
        tinggi_input = st.number_input('Masukan Nilai Tinggi')

        if st.button('Hitung Prediksi'):
            X_input = np.array([[lebar_input, panjang_input, tinggi_input]])
            price_prediction = model_regresi.predict(X_input)

            price_prediction_integer = int(price_prediction[0])

            st.subheader('Prediksi Harga Diamond :')
            st.write(f'Harga Prediksi Diamond adalah ${price_prediction_integer}')

elif selected_menu == 'Graph':
    st.subheader('Selamat Datang di Graph')
    
    fig, ax = plt.subplots()
    sns.violinplot(
        x='cut',
        y='price',
        data=data,
        palette='Set1',
        linewidth=2,
        alpha=0.7,
        ax=ax  # Pass the axes to Seaborn
    )

    ax.set_title('Violin Plot for Cut vs Price')

    ax.set_xlabel('Cut')
    ax.set_ylabel('Price')

    st.pyplot(fig)

    # Diagram 2: Joint Plot antara Harga dan Karat
    st.subheader('Joint Plot: Harga vs Karat')
    st.write(sns.jointplot(x='carat', y='price', data=data, kind='reg'))
    st.pyplot()

    # Diagram 3: Count Plot untuk Jenis Potongan (Cut)
    st.subheader('Count Plot: Jenis Potongan (Cut)')
    st.write(sns.countplot(x='cut', data=data, palette='muted'))
    st.pyplot()

    data = data.drop(data[data["x"]==0].index)
    data = data.drop(data[data["y"]==0].index)
    data = data.drop(data[data["z"]==0].index)
    data.shape