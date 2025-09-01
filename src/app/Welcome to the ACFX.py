import streamlit as st

# Page title
st.title("👋 Welcome to the ACFX Counterfactual Discovery App")

# Description
st.markdown("""
Welcome to your interactive tool for exploring **counterfactuals** using ACFX API.
---

This app is designed to help you experiment with the ACFX API in user-friendly environment.

🧪 **First Steps**  
Use the sidebar to navigate to:
- **Data Selection**: Load your dataset.
- **Classifier Selection**: Choose a predictive model.
- **Adjacency Generation**: Generate adjacency matrix and casual order. Initially, the Lingam will be used, but you can edit its evaluation result.

---

Start by selecting your data in 'Data Selection'.
""")

# Footer
st.markdown("---")
st.caption("Developed by Piotr Kubacki (piotr.kubacki@student.uj.edu.pl) · Powered by Streamlit")
