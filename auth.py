import pyrebase
import streamlit as st


# ==========================================
# FIREBASE CONFIG
# ==========================================

firebase_config = {

    "apiKey": "AIzaSyD424_nTBJ95qUIerUnw_nXpyG8S_NE2zg",

    "authDomain": "studypilot-16b6e.firebaseapp.com",

    "projectId": "studypilot-16b6e",

    "storageBucket": "studypilot-16b6e.firebasestorage.app",

    "messagingSenderId": "1043420804500",

    "appId": "1:1043420804500:web:505f6bc23333a48c23cb81",

    "databaseURL": ""

}


# ==========================================
# INITIALIZE FIREBASE
# ==========================================

firebase = pyrebase.initialize_app(
    firebase_config
)

auth = firebase.auth()


# ==========================================
# LOGIN FUNCTION
# ==========================================

def login_user(email, password):

    try:

        user = auth.sign_in_with_email_and_password(
            email,
            password
        )

        return user

    except Exception:

        return None


# ==========================================
# SIGNUP FUNCTION
# ==========================================

def signup_user(email, password):

    try:

        user = auth.create_user_with_email_and_password(
            email,
            password
        )

        return user

    except Exception:

        return None