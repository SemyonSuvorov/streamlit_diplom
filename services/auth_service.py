import streamlit as st
from supabase import create_client, Client

# Initialize Supabase client
url: str = st.secrets.supabase.url
key: str = st.secrets.supabase.anon_key
supabase: Client = create_client(url, key)

def login(email: str, password: str) -> bool:
    """Authenticate user with email and password"""
    try:
        response = supabase.auth.sign_in_with_password({
            "email": email,
            "password": password
        })
        if response.user:
            st.session_state['user'] = response.user
            return True
        return False
    except Exception as e:
        st.error(f"Login failed: {str(e)}")
        return False

def signup(email: str, password: str) -> bool:
    """Register new user"""
    try:
        response = supabase.auth.sign_up({
            "email": email,
            "password": password
        })
        if response.user:
            st.success("Registration successful! Please check your email for verification.")
            return True
        return False
    except Exception as e:
        st.error(f"Registration failed: {str(e)}")
        return False

def logout():
    """Logout current user"""
    try:
        supabase.auth.sign_out()
        st.session_state.pop('user', None)
        st.success("Logged out successfully!")
    except Exception as e:
        st.error(f"Logout failed: {str(e)}")

def get_current_user():
    """Get current authenticated user"""
    return st.session_state.get('user')

def is_authenticated():
    """Check if user is authenticated"""
    return 'user' in st.session_state 