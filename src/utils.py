import streamlit as st
def about_me():
        st.title("About the Creator")
        c1, c2 = st.columns([1,1])
        c1.markdown("""Hey! My name is **Ankit Rajput**, Mathematics and Computing student in Indian Institute Of Technology, Guwahati.
                    This app is for Face Mask Detection, I am updating this app continuously.
                    If you have any Questions or Suggestion about this app and you want to discuss it with me
                    Contact me on rajputankit72106@gmail.com""")
        c1.markdown("If you are interested :")
        c1.markdown("Github : https://github.com/ankitrajput77")
        c1.markdown("LinkedIn : https://www.linkedin.com/in/ankit-rajput892/")

        c2.image('static/Images/me.jpg')