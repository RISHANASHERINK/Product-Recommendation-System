import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Inline CSS for custom styles
css = """
<style>
body {
    background-color: #f0f8ff; /* Light blue background color */
    color: #333; /* Darker text color */
}
.main {
    background-color: #f0f8ff !important; /* Light blue background color */
}
header, .css-1v3fvcr, .css-12w0qpk {
    background-color: #f0f8ff !important; /* Light blue background color for header and footer */
}
input {
    color: #333 !important;
    background-color: #e6f7ff !important; /* Light blue background for input fields */
}
.stButton button {
    background-color: #007bff !important; /* Button color */
    color: white !important;
    border: none !important;
    padding: 10px 20px !important;
    border-radius: 5px !important;
}
.stButton button:hover {
    background-color: #0056b3 !important; /* Darker blue on hover */
}
.stTextInput label {
    color: #333 !important;
}
</style>
"""

# Inject the CSS into the Streamlit app
st.markdown(css, unsafe_allow_html=True)

# Load datasets
laptops_df = pd.read_csv('ecommerce_data.csv')
electronics_df = pd.read_csv('electronics_product.csv')

# Drop unnecessary columns and preprocess
laptops_df = laptops_df.drop(columns=['Unnamed: 0'])
laptops_df = laptops_df.dropna()
laptops_df = laptops_df.applymap(lambda s: s.lower() if type(s) == str else s)
laptops_df['combined_features'] = laptops_df['Company'] + ' ' + laptops_df['TypeName'] + ' ' + laptops_df['ScreenResolution'] + ' ' + laptops_df['Cpu'] + ' ' + laptops_df['Ram'] + ' ' + laptops_df['Memory'] + ' ' + laptops_df['Gpu'] + ' ' + laptops_df['OpSys']

electronics_df = electronics_df.dropna(subset=['name'])
electronics_df = electronics_df.applymap(lambda s: s.lower() if type(s) == str else s)
electronics_df['combined_features'] = electronics_df['name'] + ' ' + electronics_df['main_category'] + ' ' + electronics_df['sub_category']

# Initialize TF-IDF Vectorizers for each dataset
laptops_tfidf_vectorizer = TfidfVectorizer(stop_words='english')
laptops_tfidf_matrix = laptops_tfidf_vectorizer.fit_transform(laptops_df['combined_features'])
laptops_cosine_sim = cosine_similarity(laptops_tfidf_matrix, laptops_tfidf_matrix)

electronics_tfidf_vectorizer = TfidfVectorizer(stop_words='english')
electronics_tfidf_matrix = electronics_tfidf_vectorizer.fit_transform(electronics_df['combined_features'])
electronics_cosine_sim = cosine_similarity(electronics_tfidf_matrix, electronics_tfidf_matrix)

# Function to get recommendations for laptops
def get_laptop_recommendations(query):
    query_df = pd.DataFrame([query], columns=laptops_df.columns[:-2])
    query_df['combined_features'] = query_df['Company'] + ' ' + query_df['TypeName'] + ' ' + query_df['ScreenResolution'] + ' ' + query_df['Cpu'] + ' ' + query_df['Ram'] + ' ' + query_df['Memory'] + ' ' + query_df['Gpu'] + ' ' + query_df['OpSys']
    query_tfidf = laptops_tfidf_vectorizer.transform(query_df['combined_features'])
    query_sim = cosine_similarity(query_tfidf, laptops_tfidf_matrix)
    sim_scores = list(enumerate(query_sim[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Get top 5 similar laptops
    product_indices = [i[0] for i in sim_scores]
    return laptops_df.iloc[product_indices]

# Function to get recommendations for electronics products
def get_electronics_recommendations(query):
    query_df = pd.DataFrame([query])
    query_df = query_df.dropna(subset=['main_category', 'sub_category'])
    query_df['combined_features'] = query_df['main_category'] + ' ' + query_df['sub_category']
    query_tfidf = electronics_tfidf_vectorizer.transform(query_df['combined_features'])
    query_sim = cosine_similarity(query_tfidf, electronics_tfidf_matrix)
    sim_scores = list(enumerate(query_sim[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Get top 5 similar electronics products
    product_indices = [i[0] for i in sim_scores]
    return electronics_df.iloc[product_indices]

# Streamlit app
st.title('Product Recommendation System')

# User login
if 'login' not in st.session_state:
    st.session_state.login = False

if not st.session_state.login:
    username = st.text_input('Username')
    password = st.text_input('Password', type='password')
    
    if st.button('Login'):
        # Hardcoded credentials for testing
        if username == 'admin' and password == 'admin':
            st.session_state.login = True
        else:
            st.error('Invalid username or password')
else:
    st.sidebar.title('Search Filters')
    product_type = st.sidebar.selectbox('Product Type', ['Laptops', 'Electronics'])

    if product_type == 'Laptops':
        company = st.sidebar.selectbox('Company', laptops_df['Company'].unique())
        type_name = st.sidebar.selectbox('Type Name', laptops_df['TypeName'].unique())
        inches = st.sidebar.selectbox('Inches', laptops_df['Inches'].unique())
        screen_resolution = st.sidebar.selectbox('Screen Resolution', laptops_df['ScreenResolution'].unique())
        cpu = st.sidebar.selectbox('CPU', laptops_df['Cpu'].unique())
        ram = st.sidebar.selectbox('RAM', laptops_df['Ram'].unique())
        memory = st.sidebar.selectbox('Memory', laptops_df['Memory'].unique())
        gpu = st.sidebar.selectbox('GPU', laptops_df['Gpu'].unique())
        opsys = st.sidebar.selectbox('Operating System', laptops_df['OpSys'].unique())

        laptop_query = {
            'Company': company,
            'TypeName': type_name,
            'Inches': inches,
            'ScreenResolution': screen_resolution,
            'Cpu': cpu,
            'Ram': ram,
            'Memory': memory,
            'Gpu': gpu,
            'OpSys': opsys
        }

        if st.sidebar.button('Show Laptop Recommendations'):
            laptop_recommendations = get_laptop_recommendations(laptop_query)
            st.subheader('Top 5 Recommended Laptops:')
            for index, row in laptop_recommendations.iterrows():
                st.write(f"**{row['Company']} {row['TypeName']}**")
                st.write(f"*Price*: ${row['Price']}")
                st.write(f"*Specs*: {row['Cpu']}, {row['Ram']}, {row['Memory']}, {row['Gpu']}, {row['ScreenResolution']}, {row['OpSys']}")
                st.markdown("---")

    elif product_type == 'Electronics':
        main_category = st.sidebar.selectbox('Main Category', electronics_df['main_category'].unique())
        sub_category = st.sidebar.selectbox('Sub Category', electronics_df['sub_category'].unique())
        
        electronics_query = {
            'main_category': main_category,
            'sub_category': sub_category
        }

        if st.sidebar.button('Show Electronics Recommendations'):
            electronics_recommendations = get_electronics_recommendations(electronics_query)
            st.subheader('Top 5 Recommended Electronics Products:')
            for index, row in electronics_recommendations.iterrows():
                st.write(f"**{row['name']}**")
                st.write(f"*Actual Price*: ${row['actual_price']}")
                st.write(f"*Discounted Price*: ${row['discount_price']}")
                st.write(f"*Ratings*: {row['ratings']} ({row['no_of_ratings']} ratings)")
                # Assuming 'image' column contains URLs to images
                # st.image(row['image'], use_column_width=True)
                st.markdown(f"[Product Link]({row['link']})")
                st.markdown("---")

    if st.button('Logout'):
        st.session_state.login = False
