import pyrebase
import streamlit as st

from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2 import id_token
from google.auth.transport import requests


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
# EMAIL LOGIN
# ==========================================

def login_user(email, password):

    try:

        user = auth.sign_in_with_email_and_password(

            email,
            password

        )

        return user

    except Exception as e:

        print("\nEMAIL LOGIN ERROR:\n")

        print(e)

        return None


# ==========================================
# EMAIL SIGNUP
# ==========================================

def signup_user(email, password):

    try:

        user = auth.create_user_with_email_and_password(

            email,
            password

        )

        return user

    except Exception as e:

        print("\nSIGNUP ERROR:\n")

        print(e)

        return None


# ==========================================
# GOOGLE LOGIN
# ==========================================

def google_login():

    try:

        flow = InstalledAppFlow.from_client_secrets_file(

            "client_secret.json",

            scopes=[

                "openid",

                "https://www.googleapis.com/auth/userinfo.email",

                "https://www.googleapis.com/auth/userinfo.profile",

                "https://www.googleapis.com/auth/calendar"

            ]

        )

        credentials = flow.run_local_server(

            port=0

        )

        request_object = requests.Request()

        user_info = id_token.verify_oauth2_token(

            credentials.id_token,

            request_object

        )

        print("\nGOOGLE LOGIN SUCCESS\n")

        print(user_info)

        return user_info

    except Exception as e:

        print("\nGOOGLE LOGIN ERROR:\n")

        print(e)

        return None