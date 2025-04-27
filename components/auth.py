import streamlit as st
from services.auth_service import login, signup, logout, is_authenticated

def show_auth_form():
    """Display authentication form"""
    tab1, tab2 = st.tabs(["Вход", "Регистрация"])
    
    with tab1:
        st.subheader("Вход")
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Пароль", type="password", key="login_password")
        if st.button("Войти"):
            if login(email, password):
                st.rerun()
    
    with tab2:
        st.subheader("Регистрация")
        email = st.text_input("Email", key="signup_email")
        password = st.text_input("Пароль", type="password", key="signup_password")
        confirm_password = st.text_input("Подтвердите пароль", type="password", key="confirm_password")
        if st.button("Зарегистрироваться"):
            if password == confirm_password:
                if signup(email, password):
                    st.rerun()
            else:
                st.error("Пароли не совпадают!")

def show_logout_button():
    """Display logout button"""
    if st.button("Выйти"):
        logout()
        st.rerun() 