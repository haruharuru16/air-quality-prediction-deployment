import streamlit as st
from helper import load_model, to_dataframe, get_summary, data_preprocessing, get_prediction

st.set_page_config(layout='wide')


def main():
    st.title('Air Quality Prediction Using Random Forest')

    # load the trained model
    classifier = load_model('rf_final_model.pkl')

    # load the normalizer
    robust_normalizer = load_model('robust_train_normalizer.pkl')

    # input the data
    # pm10, so2, co, o3, no2
    col1, col2 = st.columns([1, 1])
    with col1:
        pm10 = st.number_input('PM10 Concentration')
        so2 = st.number_input('SO2 Concentration')
        co = st.number_input('CO Concentration')

    with col2:
        o3 = st.number_input('O3 Concentration')
        no2 = st.number_input('NO2 Concentration')

    # when the 'Predict' is clicked, make the prediction and store it
    if st.button('Predict'):
        # convert the data into dataframe
        air_quality_data = to_dataframe(pm10, so2, co, o3, no2)

        # preprocess the data
        data_prep = data_preprocessing(air_quality_data, robust_normalizer)

        # get prediction result
        result = get_prediction(data_prep, classifier)

        # show the data summary and the predicted result
        summary, predicted = st.columns([1, 2])
        with summary:
            st.markdown('**Prediction Summary**')

            # show data summary
            summary_data = get_summary(air_quality_data)
            st.write(summary_data)

        with predicted:
            st.markdown('**Prediction Result**')

            # show the predicted result
            st.write(f'Predicted Air Quality Level : **{result}**')


if __name__ == '__main__':
    main()
