import streamlit as st                                  # Importing Streamlit library for building web apps
from streamlit_option_menu import option_menu           # Importing option_menu from streamlit_option_menu for dropdown menus
import pandas as pd                                     # Importing pandas for data manipulation
import plotly.express as px 
import time                                             # Importing time module for time-related functions
import matplotlib.pyplot as plt                         # Importing matplotlib.pyplot for plotting
import seaborn as sns                                   # Importing seaborn for statistical data visualization
import numpy as np
import joblib



image_logo = "https://png.pngtree.com/png-vector/20220329/ourmid/pngtree-warning-h1n1-sign-sick-veterinary-distraught-vector-png-image_5323144.jpg"


st.set_page_config(page_title="H1N1 Vaccine Usage Analysis & Prediction", page_icon=image_logo, layout="wide")

background_color = "#0077b6"
st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {background_color};
    }}
    </style>
    """,
    unsafe_allow_html=True
)




st.markdown("""
<style>.big-title { color: #caf0f8; font-size: 36px; text-align: center; font-weight: bold; }</style>
<p class="big-title">H1N1 Vaccine Usage Analysis & Prediction</p>
""", unsafe_allow_html=True)




def classificationprediction(input_data):
    with open(r"C:\My Folder\Tuts\Python\Project\Project 5 - Final Project\Vaccine Usage Prediction\Vaccine Usage Prediction\Vaccine_Usage_Prediction_Model.pkl", "rb") as f:
        model = joblib.load(f)
        # model predict the vaccine usage based on user input
        y_predict = model.predict(input_data)

    return y_predict



def show_toast(message, color):
  alert =   st.markdown(f"""
        <div style="position: fixed; bottom: 10%; right: 10%; background-color: {color}; color: white; padding: 10px; border-radius: 5px; z-index: 1000;">
            {message}
        </div>
    """, unsafe_allow_html=True)
  
  time.sleep(3)
  alert.empty()


col1,col2,col3 = st.columns([1.0,1.9,1.0])

with col1:
    pass

with col2:
    # Creating menu using the streamlit_option_menu library
    Selected_Option = option_menu(
        menu_title = None,                                                                              # Title of the menu (None in this case)
        options=["Home","Analysis","Prediction"],                                                       # List of menu options
        default_index= 0,                                                                               # Index of the default selected option
        icons =["house-heart-fill", "pie-chart-fill", "puzzle-fill"],     # Icons for each option
        orientation="horizontal",                                                                       # Orientation of the menu (horizontal or vertical)
        styles={
        "container": {"background-color": "#00b4d8", 'width': '625px', 'color': 'yellow'}, # Styles for the menu container
        "icon": {"color": "#caf0f8", "font-size": "17px"},                                                # Styles for the icons
        "nav-link": { "--hover-color": "#94d2bd","color": "90e0ef", 'width': '200px', 
                        "text-align":"center","padding":"5px 0",
                        "border-bottom":"1px solid transparent","transition":"border-bottom 0.3 ease","font-size":"15px", "margin": "0px"},    # Styles for the menu links
        "nav-link-selected": {"background-color": "#f07167",  "border-bottom":"5px solid #0077b6","color":"#caf0f8", 'width': '200px', 'icon-color':'#caf0f8'}       # Styles for the selected menu link
        }
    )

with col3:
    pass

# Check if the selected option is "Home"
if Selected_Option == "Home":
    st.write('')
    st.write('')
    st.write('')


    st.markdown("""<h1 style='color: white; font-size: 40px; font-weight: bold;'>H1N1 Vaccine Prediction App</h1>""",unsafe_allow_html=True)
    st.write('')
    st.markdown("""<div style='color: white; font-size: 20px; font-weight: bold;'>Welcome to the H1N1 Vaccine Prediction App!</div>""",unsafe_allow_html=True)
    st.write('')
    st.markdown("""<div style='color: white; font-size: 15px;'>This app helps predict the likelihood of someone taking the H1N1 flu vaccine based on their characteristics and attitudes. This information can be valuable for public health officials and policymakers in targeting vaccination campaigns more effectively.</div>""",unsafe_allow_html=True)
    st.write('')
    st.write('')

    st.markdown(
    """
    <div style='color: white; font-size: 20px; font-weight: bold;'>What can you do here?</div>
    <ul style='color: white; font-size: 16px;'>
        <li><strong>Learn about H1N1:</strong> Get a brief overview of the H1N1 flu virus and the importance of vaccination.</li>
        <li><strong>Explore Data:</strong> Discover trends and insights from the data used to build the prediction model. You can use interactive charts and filters to gain a deeper understanding of factors influencing vaccine acceptance.</li>
        <li><strong>Predict Vaccine Usage:</strong> Input your own answers to a series of questions related to demographics, behaviors, and perceptions. The app will use our trained machine learning model to predict the probability of you taking the H1N1 flu vaccine.</li>
        <li><strong>Understand the Model:</strong> Learn more about the model used for prediction, including its limitations and how it was developed.</li>
    </ul>
    """,
    unsafe_allow_html=True
    )
    st.write('')
    st.markdown(
    """
    <div style='color: white; font-size: 20px; font-weight: bold;'>Additional Information:</div>
    <ul style='color: white; font-size: 16px;'>
        <li><strong>Learn about H1N1:</strong> This app is for informational purposes only and should not be used as a substitute for professional medical advice.</li>
        <li><strong>Explore Data:</strong> The data used to train the model may not be completely representative of the entire population.</li>
    </ul>
    """,
    unsafe_allow_html=True
    )
    
    st.markdown("""<div style='color: white; font-size: 15px;'>We encourage you to explore the app and learn more about H1N1 vaccination!</div>""",unsafe_allow_html=True)


if Selected_Option == "Analysis":
    st.write('')
    st.title('H1N1 Survey Data Visualization')
    
    df = pd.read_csv(r"C:\My Folder\Tuts\Python\Project\Project 5 - Final Project\Vaccine Usage Prediction\Dataset\vaccine_dataset.csv")

    col1, col2, col3, col4, col5 = st.columns([3,3,3,3,3])
    with col1:
        h1n1_worry_dropdown = st.selectbox('Worry about H1N1 Flu', options = ['All', 'Not very worried', 'Very worried', 'Somewhat worried', 'Not worried at all'], index = 0,)
    with col2:
        age_bracket_dropdown = st.selectbox('Age Bracket', options = ['All', '18 - 34 Years', '35 - 44 Years', '45 - 54 Years', '55 - 64 Years', '65+ Years'], index = 0,)
    with col3:
        qualification_dropdown = st.selectbox('Education Level', options = ['All', '< 12 Years', '12 Years', 'College Graduate', 'Some College'], index = 0,)
    with col4:
        h1n1_awareness_dropdown = st.selectbox('Knowledge about H1N1 Flu', options = ['All', 'No Knowledge', 'Good Knowledge', 'Little Knowledge'], index = 0,)
    with col5:
        census_msa_dropdown = st.selectbox('Metropolitan Statistical Area', options = ['All', 'Non-MSA', 'MSA, Not Principle  City', 'MSA, Principle City'], index = 0,)

    # Apply filters to the DataFrame
    if h1n1_worry_dropdown != 'All':
        df = df[df['h1n1_worry'] == h1n1_worry_dropdown]

    if age_bracket_dropdown != 'All':
        df = df[df['age_bracket'] == age_bracket_dropdown]

    if qualification_dropdown != 'All':
        df = df[df['qualification'] == qualification_dropdown]

    if h1n1_awareness_dropdown != 'All':
        df = df[df['h1n1_awareness'] == h1n1_awareness_dropdown]

    if census_msa_dropdown != 'All':
        df = df[df['census_msa'] == census_msa_dropdown]


    data = df.copy()

    chart_type = st.sidebar.selectbox('Select Chart Type:', ['Univariate', 'Bivariate', 'Multivariate'])

    # Univariate Charts
    if chart_type == 'Univariate':
        univariate_chart_type = st.sidebar.selectbox('Select Univariate Chart Type:', ['Bar Chart', 'Pie Chart', 'Count Plot'])
        chart_width = st.sidebar.slider('Select Chart Width:', 400, 800, 600)
        chart_height = st.sidebar.slider('Select Chart Height:', 400, 800, 600)

        # Bar Chart
        if univariate_chart_type == 'Bar Chart':
            st.header('Bar Chart')
            column = st.sidebar.selectbox('Select column for Bar Chart:', data.columns)
            chart_data = data[column].value_counts().reset_index()
            chart_data.columns = [column, 'Count']
            fig = px.bar(chart_data, x=column, y='Count', text='Count', width=chart_width, height=chart_height)
            fig.update_traces(texttemplate='%{text}', textposition='outside')
            st.plotly_chart(fig)

        # Pie Chart
        elif univariate_chart_type == 'Pie Chart':
            st.header('Pie Chart')
            column = st.sidebar.selectbox('Select column for Pie Chart:', data.columns)
            vaccine_distribution = data[column].value_counts().reset_index()
            vaccine_distribution.columns = [column, 'Count']
            fig_pie = px.pie(vaccine_distribution, values='Count', names=column, width=chart_width, height=chart_height, hole=0.5)
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie)

        # Count Plot
        elif univariate_chart_type == 'Count Plot':
            st.header('Count Plot')
            column = st.sidebar.selectbox('Select column for Count Plot:', data.columns)
            fig, ax = plt.subplots(figsize=((chart_width/100)-2, (chart_height/100)-2))
            sns.countplot(data=data, x=column, ax=ax)
            for p in ax.patches:
                ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='baseline')
            st.pyplot(fig)


    # Bivariate Charts
    elif chart_type == 'Bivariate':
        try:
            st.header('Bivariate Charts')
            # chart_choice = st.sidebar.selectbox('Select Bivariate Chart Type:', ['Scatter Plot', 'Line Plot', 'Bar Chart'])
            chart_choice = st.sidebar.selectbox('Select Bivariate Chart Type:', ['Scatter Plot', 'Bar Chart', 'Heat Map', 'Line Plot'])
            chart_width = st.sidebar.slider('Select Chart Width:', 400, 800, 600)
            chart_height = st.sidebar.slider('Select Chart Height:', 400, 800, 600)

            # Scatter Plot
            if chart_choice == 'Scatter Plot':
                # x_column = st.sidebar.selectbox('Select X column:', data.columns)
                # y_column = st.sidebar.selectbox('Select Y column:', data.columns)
                x_column = st.sidebar.selectbox('Select X column:', data.columns, index=0)  # First column by iloc[0]
                y_column = st.sidebar.selectbox('Select Y column:', data.columns, index=1)  # Second column by iloc[1]

                grouped_data = data.groupby([x_column, y_column]).size().reset_index(name='counts_new')
                fig = px.scatter(grouped_data, x=x_column, y=y_column, size='counts_new', hover_data=['counts_new'], width=chart_width, height=chart_height)
                # st.plotly_chart(fig)
                
                # fig = px.scatter(data, x=x_column, y=y_column, width=chart_width, height=chart_height)
                fig.update_traces(textposition='top center')
                st.plotly_chart(fig)

            # Line Plot
            elif chart_choice == 'Line Plot':
                # x_column = st.sidebar.selectbox('Select X column:', data.columns, index = 0)
                # y_column = st.sidebar.selectbox('Select Y column:', data.columns, index = 1)

                integer_columns = data.select_dtypes(include=['int64', 'int32']).columns
                non_integer_columns = data.select_dtypes(exclude=['int64', 'int32']).columns

                if len(non_integer_columns) > 0:
                    x_column = st.sidebar.selectbox('Select X column:', non_integer_columns)
                else:
                    st.warning("No non-integer columns available for X column.")                

                if len(integer_columns) > 0:
                    y_column = st.sidebar.selectbox('Select Y column:', integer_columns, index=1)
                else:
                    st.warning("No integer columns available for Y column.")

                color = st.sidebar.selectbox('Select Color:', data.columns, index = 3)
                grouped_data = data.groupby([x_column, color])[y_column].mean().reset_index()
                fig = px.line(
                    grouped_data, x=x_column, y=y_column, color=color,  # Use color to differentiate groups
                    width=chart_width, height=chart_height
                )
                fig.update_traces(textposition='top center')
                st.plotly_chart(fig)
            
            # Bar Chart
            elif chart_choice == 'Bar Chart':
                x_column = st.sidebar.selectbox('Select X column:', data.columns, index=0)
                y_column = st.sidebar.selectbox('Select Y column:', data.columns, index=1)
                # fig = px.bar(data, x=x_column, y=y_column, text=y_column, width=chart_width, height=chart_height)
                # fig.update_traces(texttemplate='%{text}', textposition='outside')
                # st.plotly_chart(fig)

                grouped_data = data.groupby([x_column, y_column]).size().reset_index(name='counts_new')
                # Plot using 'counts'
                fig = px.bar(grouped_data, x=x_column, y='counts_new', color=y_column, text='counts_new', width=chart_width, height=chart_height)
                # Update text position
                fig.update_traces(texttemplate='%{text}', textposition='outside')
                # Display the plot
                st.plotly_chart(fig)

            elif chart_choice == 'Heat Map':
                x_column = st.sidebar.selectbox('Select X column:', data.columns, index=0)
                y_column = st.sidebar.selectbox('Select Y column:', data.columns, index=1)
                pivot_table = data.groupby([x_column, y_column]).size().unstack(fill_value=0)
                fig = px.imshow(pivot_table, text_auto=True)
                st.plotly_chart(fig)
        except:
            # st.error('Same variable selection for both x and y axis will not be allowed. Use Different variables for both x and y axis')
            st.markdown(
                """<p style="background-color:red; color:white; font-size:20px; padding:10px; border-radius:5px;">
                Same variable selection for both x and y axis will not be allowed. Use different variables for both x and y axis.</p>""",
                unsafe_allow_html=True
            )


    elif chart_type == 'Multivariate':
        st.header('Multivariate Charts')
        # Multivariate scatter plot with color and size encoding
        x_column = st.sidebar.selectbox('Select X column:', data.columns, index = 0)
        y_column = st.sidebar.selectbox('Select Y column:', data.columns, index = 1)
        color_column = st.sidebar.selectbox('Select color column:', data.columns, index = 2)
        size_column = st.sidebar.selectbox('Select size column:', data.columns, index = 3)
        chart_width = st.sidebar.slider('Select Chart Width:', 400, 800, 600)
        chart_height = st.sidebar.slider('Select Chart Height:', 400, 800, 600)
        try:
            grouped_data = data.groupby([x_column, y_column, color_column, size_column]).size().reset_index(name='counts')
            fig = px.scatter(
                grouped_data, x=x_column, y=y_column, 
                color=color_column, size='counts',  # Use the counts as the size
                width=chart_width, height=chart_height,
                hover_data=grouped_data.columns
            )

            st.plotly_chart(fig)        
        except:
            # st.error('Please select different columns to display chart for Multivarite Analysis')
            st.markdown(
                """<p style="background-color:red; color:white; font-size:20px; padding:10px; border-radius:5px;">
                Same variable selection for Multivariate will not be allowed. Please use different variables.</p>""",
                unsafe_allow_html=True
            )



    st.sidebar.header('Additional Settings')

    if st.sidebar.checkbox('Show Pairplot'):
        st.header('Pairplot')

        # You can optionally let the user select a hue for categorical coloring
        hue_column = st.sidebar.selectbox('Select hue column (optional):', [None] + list(data.columns))

        # If the user selects a hue, apply it
        if hue_column and hue_column != 'None':
            fig = sns.pairplot(data, kind='scatter', plot_kws={'alpha': 0.5}, hue=hue_column)
        else:
            fig = sns.pairplot(data, kind='scatter', plot_kws={'alpha': 0.5})

        st.pyplot(fig)


if Selected_Option == "Prediction":
    cola, colb = st.columns([0.65,5])
    with cola:
        pass
    with colb:
        st.write('')
        st.write('')
        st.write('')
        st.markdown("""<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">""", unsafe_allow_html=True)
        # st.markdown("""<p style="color:white; font-size:30px; font-weight:bold;">Vaccine Usage Prediction Form</p>""", unsafe_allow_html=True)
        st.markdown("""<p style="color:white; font-size:30px; font-weight:bold;">Vaccine Usage Prediction Form <i style='font-size:24px; color:white;' class='fas'>&#xf48e;</i></p>""", unsafe_allow_html=True)
        st.markdown("""<hr style="border: 1px solid white; width: 86%;">""", unsafe_allow_html=True)
        st.write('')

    col1, col2, col3, col4, col5 = st.columns([1,3,0.5,3,1])
    with col1:
        pass
    with col2:
        h1n1_worry = st.selectbox('Worry about H1N1 Flu', ['Not worried at all', 'Not very worried', 'Somewhat worried', 'Very worried'], key = 'h1n1_worry')
        antiviral_medication = st.selectbox('Taken Antiviral Medication', ['Yes','No'], key = 'antiviral_medication')
        bought_face_mask = st.selectbox('Bought Face Mask', ['Yes','No'], key = 'bought_face_mask')
        avoid_large_gatherings = st.selectbox('Avoided Large Gatherings', ['Yes','No'], key = 'avoid_large_gatherings')
        avoid_touch_face = st.selectbox('Avoids Touching Face', ['Yes','No'], key = 'avoid_touch_face')
        dr_recc_seasonal_vacc = st.selectbox('Doctor Recommended Seasonal Flu Vaccine', ['Yes','No'], key = 'dr_recc_seasonal_vacc')
        cont_child_undr_6_mnths = st.selectbox('Contact with Child under 6 Months', ['Yes','No'], key = 'cont_child_undr_6_mnth')
        has_health_insur = st.selectbox('Has Health Insurance', ['Yes','No'], key = 'has_health_insur')
        is_h1n1_risky = st.selectbox('Perceived Risk of H1N1 Flu', ['Thinks it is not very low risk', 'Thinks it is somewhat low risk', 'Don’t know if it is risky or not', 'Thinks it is a somewhat high risk', 'Thinks it is very highly risky'], key = 'is_h1n1_risky')
        is_seas_vacc_effective = st.selectbox('Belief in Seasonal Vaccine Effectiveness', ['Thinks not effective at all', 'Thinks it is not very effective', "Doesn't know if it is effective or not", 'Thinks it is somewhat effective', 'Thinks it is highly effective'], key = 'is_seas_vacc_effective')
        sick_from_seas_vacc = st.selectbox('Worry about Getting Sick from Seasonal Vaccine', ['Respondent not worried at all', 'Respondent is not very worried', "Doesn't know", 'Respondent is somewhat worried', 'Respondent is very worried'], key = 'sick_from_seas_vacc')
        qualification = st.selectbox('Education Level', ['< 12 Years', '12 Years', 'College Graduate', 'Some College'], key = 'qualification')
        sex = st.selectbox('Sex', ['Female', 'Male'], key = 'sex')
        marital_status = st.selectbox('Marital Status', ['Not Married', 'Married'], key = 'marital_status')
        employment = st.selectbox('Employment Status', ['Not In Labor Force', 'Employed', 'Unemployed'], key = 'employment')
        no_of_adults = st.selectbox('Number of Adults in Household', [0,1,2,3], key = 'no_of_adults')

    with col3:
        pass

    with col4:
        h1n1_awareness = st.selectbox('Knowledge about H1N1 Flu', ['No Knowledge', 'Little Knowledge', 'Good Knowledge'], key = 'h1n1_awareness')
        contact_avoidance = st.selectbox('Avoided Close Contact with Sick People', ['Yes','No'], key = 'contact_avoidance')
        wash_hands_frequently = st.selectbox('Frequently Washes Hands', ['Yes','No'], key = 'wash_hands_frequently')
        reduced_outside_home_cont = st.selectbox('Reduced Contact Outside Home', ['Yes','No'], key = 'reduced_outside_home_cont')
        dr_recc_h1n1_vacc = st.selectbox('Doctor Recommended H1N1 Vaccine', ['Yes','No'], key = 'dr_recc_h1n1_vacc')
        chronic_medic_condition = st.selectbox('Has Chronic Medical Condition', ['Yes','No'], key = 'chronic_medic_condition')
        is_health_worker = st.selectbox('Health Worker', ['Yes','No'], key = 'is_health_worker')
        is_h1n1_vacc_effective = st.selectbox('Belief in H1N1 Vaccine Effectiveness', ['Thinks not effective at all', 'Thinks it is not very effective', "Doesn't know if it is effective or not", 'Thinks it is somewhat effective', 'Thinks it is highly effective'], key = 'is_h1n1_vacc_effective')
        sick_from_h1n1_vacc = st.selectbox('Worry about Getting Sick from H1N1 Vaccine', ['Respondent not worried at all', 'Respondent is not very worried', "Doesn't know", 'Respondent is somewhat worried', 'Respondent is very worried'], key = 'sick_from_h1n1_vacc')
        is_seas_risky = st.selectbox('Perceived Risk of Seasonal Flu', ['Thinks it is not very low risk', 'Thinks it is somewhat low risk', "Doesn't know if it is risky or not", 'Thinks it is somewhat high risk', 'Thinks it is very highly risky'], key = 'is_seas_risky')
        age_bracket = st.selectbox('Age Bracket', ['55 - 64 Years', '35 - 44 Years', '18 - 34 Years', '65+ Years', '45 - 54 Years'], key = 'age_bracket')
        race = st.selectbox('Race', ['White', 'Black', 'Other Or Multiple', 'Hispanic'], key = 'race')
        income_level = st.selectbox('Income Level', ['Below Poverty', '<= $75,000, Above Poverty', '> $75,000'], key = 'income_level')
        housing_status = st.selectbox('Housing Status', ['Own', 'Rent'], key = 'housing_status')
        census_msa = st.selectbox('Residence in Metropolitan Statistical Area', ['Non-MSA', 'MSA, Not Principle  City', 'MSA, Principle City'], key = 'census_msa')
        no_of_children = st.selectbox('Number of Children in Household', [0,1,2,3], key = 'no_of_children')

    with col5:
        pass

    
    h1n1_worry_encoded = {'Not worried at all': 0, 'Not very worried': 1, 'Somewhat worried': 2, 'Very worried': 3}
    h1n1_awareness_encoded = {'No Knowledge': 0, 'Little Knowledge': 1, 'Good Knowledge': 2}
    antiviral_medication_encoded = {'No': 0, 'Yes': 1}
    contact_avoidance_encoded = {'No': 0, 'Yes': 1}
    bought_face_mask_encoded = {'No': 0, 'Yes': 1}
    wash_hands_frequently_encoded = {'No': 0, 'Yes': 1}
    avoid_large_gatherings_encoded = {'No': 0, 'Yes': 1}
    reduced_outside_home_cont_encoded = {'No': 0, 'Yes': 1}
    avoid_touch_face_encoded = {'No': 0, 'Yes': 1}
    dr_recc_h1n1_vacc_encoded = {'No': 0, 'Yes': 1}
    dr_recc_seasonal_vacc_encoded = {'No': 0, 'Yes': 1}
    chronic_medic_condition_encoded = {'No': 0, 'Yes': 1}
    cont_child_undr_6_mnths_encoded = {'No': 0, 'Yes': 1}
    is_health_worker_encoded = {'No': 0, 'Yes': 1}
    has_health_insur_encoded = {'No': 0, 'Yes': 1}
    is_h1n1_vacc_effective_encoded = {'Thinks not effective at all': 1, 'Thinks it is not very effective': 2, "Doesn't know if it is effective or not": 3, 'Thinks it is somewhat effective': 4, 'Thinks it is highly effective': 5}
    is_h1n1_risky_encoded = {'Thinks it is not very low risk': 1, 'Thinks it is somewhat low risk': 2, 'Don’t know if it is risky or not': 3, 'Thinks it is a somewhat high risk': 4, 'Thinks it is very highly risky': 5}
    sick_from_h1n1_vacc_encoded = {'Respondent not worried at all': 1, 'Respondent is not very worried': 2, "Doesn't know": 3, 'Respondent is somewhat worried': 4, 'Respondent is very worried': 5}
    is_seas_vacc_effective_encoded = {'Thinks not effective at all': 1, 'Thinks it is not very effective': 2, "Doesn't know if it is effective or not": 3, 'Thinks it is somewhat effective': 4, 'Thinks it is highly effective': 5}
    is_seas_risky_encoded = {'Thinks it is not very low risk': 1, 'Thinks it is somewhat low risk': 2, "Doesn't know if it is risky or not": 3, 'Thinks it is somewhat high risk': 4, 'Thinks it is very highly risky': 5}
    sick_from_seas_vacc_encoded = {'Respondent not worried at all': 1, 'Respondent is not very worried': 2, "Doesn't know": 3, 'Respondent is somewhat worried': 4, 'Respondent is very worried': 5}
    age_bracket_encoded = {'18 - 34 Years': 0, '35 - 44 Years': 1, '45 - 54 Years': 2, '55 - 64 Years': 3, '65+ Years': 4}
    qualification_encoded = {'12 Years': 0, '< 12 Years': 1, 'College Graduate': 2, 'Some College': 3}
    race_encoded = {'Black': 0, 'Hispanic': 1, 'Other Or Multiple': 2, 'White': 3}
    sex_encoded = {'Female': 0, 'Male': 1}
    income_level_encoded = {'<= $75,000, Above Poverty': 0, '> $75,000': 1, 'Below Poverty': 2}
    marital_status_encoded = {'Married': 0, 'Not Married': 1}
    housing_status_encoded = {'Own': 0, 'Rent': 1}
    employment_encoded = {'Employed': 0, 'Not In Labor Force': 1, 'Unemployed': 2}
    census_msa_encoded = {'MSA, Not Principle  City': 0, 'MSA, Principle City': 1, 'Non-MSA':2}


    cola, colb = st.columns([0.65,5])
    with cola:
        pass
    with colb:
        st.write('')
        st.markdown("""<hr style="border: 1px solid white; width: 86%;">""", unsafe_allow_html=True)

    cola, colb = st.columns([4,5])
    with cola:
        pass
    with colb:

        if st.button("Predict Usage"):

            input_data = np.array([[int(h1n1_worry_encoded[h1n1_worry]),             
                                    int(h1n1_awareness_encoded[h1n1_awareness]),
                                    int(antiviral_medication_encoded[antiviral_medication]),
                                    int(contact_avoidance_encoded[contact_avoidance]),
                                    int(bought_face_mask_encoded[bought_face_mask]),
                                    int(wash_hands_frequently_encoded[wash_hands_frequently]),
                                    int(avoid_large_gatherings_encoded[avoid_large_gatherings]),
                                    int(reduced_outside_home_cont_encoded[reduced_outside_home_cont]),
                                    int(avoid_touch_face_encoded[avoid_touch_face]),
                                    int(dr_recc_h1n1_vacc_encoded[dr_recc_h1n1_vacc]),
                                    int(dr_recc_seasonal_vacc_encoded[dr_recc_seasonal_vacc]),
                                    int(chronic_medic_condition_encoded[chronic_medic_condition]),
                                    int(cont_child_undr_6_mnths_encoded[cont_child_undr_6_mnths]),
                                    int(is_health_worker_encoded[is_health_worker]),
                                    int(has_health_insur_encoded[has_health_insur]),
                                    int(is_h1n1_vacc_effective_encoded[is_h1n1_vacc_effective]),
                                    int(is_h1n1_risky_encoded[is_h1n1_risky]),
                                    int(sick_from_h1n1_vacc_encoded[sick_from_h1n1_vacc]),
                                    int(is_seas_vacc_effective_encoded[is_seas_vacc_effective]),
                                    int(is_seas_risky_encoded[is_seas_risky]),
                                    int(sick_from_seas_vacc_encoded[sick_from_seas_vacc]),
                                    int(age_bracket_encoded[age_bracket]),
                                    int(qualification_encoded[qualification]),
                                    int(race_encoded[race]),
                                    int(sex_encoded[sex]),
                                    int(income_level_encoded[income_level]),
                                    int(marital_status_encoded[marital_status]),
                                    int(housing_status_encoded[housing_status]),
                                    int(employment_encoded[employment]),
                                    int(census_msa_encoded[census_msa]),
                                    int(no_of_adults),
                                    int(no_of_children)
                                    ]])



            usage_prediction = classificationprediction(input_data)
            st.write('')
            if usage_prediction == 0:
                show_toast('🙁 Respondent did not receive the H1N1 Vaccine', 'Red')
            else:
                show_toast('🤩 Respondent received the H1N1 Vaccine', 'Green')





