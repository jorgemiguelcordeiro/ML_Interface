# ML_Interface

This interface on streamlit was developed on behalf of the Machine learning project (check github repository). 
The integration of Streamlit provides a user-friendly platform for collecting and processing 
input data, applying the preprocessing pipeline in real time to ensure accurate predictions. 
However, during implementation, the application crashed upon clicking the Predict button, 
likely due to resource limitations. Streamlit can struggle with large datasets or intensive 
computations on local machines. Using frameworks like Flask, Dash, or deploying on a 
cloud platform with scalable resources could address this issue. To verify that the problem 
is not in the code, we tested a simpler interface using Jupyter Notebook's widgets, which 
worked flawlessly, confirming that the issue is related to Streamlit's resource handling. You 
can check the streamlit here: https://claiminjurytypepredictionapp.streamlit.app
