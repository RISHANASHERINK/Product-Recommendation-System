# Product Recommendation System 

This project implements a product recommendation system using machine learning models. 
It supports recommendations for laptops and electronics products based on user search queries. 
The application is built using Streamlit for the web interface.

## Features

- Recommends laptops based on technical specifications.
- Recommends electronics products based on categories.
- Simple and interactive web interface.

## Installation

### Prerequisites

- [Python](https://www.python.org/downloads/) (version 3.7 or higher)
- [Git](https://git-scm.com/downloads)
- [Visual Studio Code (VSCode)](https://code.visualstudio.com/Download)

### Steps

1. **Clone the repository:**
    
    git clone https://github.com/RISHANASHERINK/product-recommendation-system.git
    cd product-recommendation-system
    
4. **Install the required packages:**

   
    pip install -r requirements.txt
    pandas
    scikit-learn
    streamlit
    numpy

   

## Running the Application in VSCode

1. **Open the project folder in VSCode:**

    Open VSCode and navigate to `File > Open Folder...`, then select the `product-recommendation-system` folder.

2. **Open the terminal in VSCode:**

4. **Run the Streamlit app:**

    streamlit run app.py
   
5. **Open your web browser and go to:**

    http://localhost:8501

## Dataset

- **ecommerce_data.csv**: Contains laptop specifications.
- **electronics_product.csv**: Contains electronics product details.

## How It Works

- **Laptop Recommendation**: Based on the technical specifications provided by the user, the system recommends laptops with similar features.
- **Electronics Recommendation**: Based on the main category and sub-category provided by the user, the system recommends similar electronics products.

 
