# Import Important Libraries

import joblib
import streamlit as st 
from PIL import Image
import sklearn

# Load Model

model = joblib.load('model.pkl')

# Load Vectorizer

vectorizer = joblib.load('vectorizer.pkl')

# Load Image

image = Image.open('img.jpg')

# Streamlit Function For Building Button & app.

def main():
    st.image(image, width=650)
    st.title('Mental Health ChatBot')
    
    # HTML Styling for title and input box
    html_temp = '''
    <div style='background-color:red; padding:12px'>
    <h1 style='color:  #000000; text-align: center;'>Mental Health ChatBot Machine Learning Model</h1>
    </div>
    '''
    st.markdown(html_temp, unsafe_allow_html=True)
    # Add copyright notice
    copyright_text = '''
    <div style=width: 20%; text-align: center; padding: 12px; background-color: black'>
    <p style='font-size: 14px;color:red'>© 2024 PAWAN YADAV (AI Engineer). All Rights Reserved.</p></div>'''
    st.markdown(copyright_text, unsafe_allow_html=True)
    # Input Box for user queries
    user_input = st.text_input('Enter Your Query')

    # Store prediction result
    result = ''

    # Predict when the button is clicked
    if st.button('Predict'):
        if user_input.strip() != "":
            # Process the input and display results
            result = prediction(user_input)
            new_result=process_prediction(result)
            formatted_result = new_result.replace("\n", "<br>")  # Replace newlines with <br> for HTML rendering
            
            temp = f'''
            <div style='background-color:purple; padding:8px'>
            <h1 style='color: black; text-align: center;'>{formatted_result}</h1>
            </div>
            '''
            st.markdown(temp, unsafe_allow_html=True)
        else:
            st.error("Please enter a valid input.")


# Prediction Function to predict from the model
def prediction(user_input):
    # Vectorizing the input
    new_text_vectorized = vectorizer.transform([user_input])  # Ensure input is wrapped in a list
    predict_proba = model.predict_proba(new_text_vectorized)  # Get the probability for each class
    
    # Define a threshold for prediction confidence
    threshold = 0.5  # Set a threshold (e.g., 20% confidence)
    max_prob = max(predict_proba[0])
    
    # If the highest probability is below the threshold, return 'No mental health issue'
    if max_prob < threshold:
        return 'NO'
    
    # Get the prediction class (based on the highest probability)
    predicted_class = predict_proba[0].argmax()  # Find the index of the max probability

    # Map the predicted class to the corresponding mental health issue
    if predicted_class == 0:
        return "Stress"
    elif predicted_class == 1:
        return "Depression"
    elif predicted_class == 2:
        return "Bipolar disorder"
    elif predicted_class == 3:
        return "Personality disorder"
    elif predicted_class == 4:
        return "Anxiety"
    else:
        return "NO"
    
# Process the prediction and format response
def process_prediction(prediction):
    if prediction == "Stress":
        return (
            "It seems like you're experiencing stress. Remember, it's important to take care of your mental health.\n"
            "Here are some tips:\n\n"
            "- **Take deep breaths or try mindfulness exercises.**\n"
            "- **Talk to someone you trust about your feelings.**\n"
            "- **Make sure you're getting enough rest and proper nutrition.**\n"
            "- **Consider seeking support from a mental health professional if needed.**\n\n"
            "You're not alone, and there are people who care and want to help you. Be happy! 😊"
        )
    elif prediction == "Depression":
        return (
            "It seems like you're experiencing symptoms of depression. Please know that you are not alone, and help is available.\n"
            "Here are some suggestions that may help:\n\n"
            "- **Talk to a trusted friend or family member about how you're feeling.**\n"
            "- **Try to maintain a daily routine and include activities that you enjoy.**\n"
            "- **Consider reaching out to a mental health professional for guidance and support.**\n"
            "- **Engage in light physical activity like walking or yoga, which can boost your mood.**\n"
            "- **Practice self-compassion; take one step at a time toward feeling better.**\n\n"
            "Remember, seeking help is a sign of strength, and there are people who care deeply about your well-being. Be happy! 😊"
        )
    elif prediction == "Bipolar disorder":
        return (
            "It seems like you're experiencing symptoms associated with Bipolar Disorder. This condition can lead to significant emotional highs "
            "(mania or hypomania) and lows (depression). Please remember that Bipolar Disorder is manageable with the right support and treatment.\n\n"
            "Here are a few suggestions that may help:\n"
            "- **Seek Professional Help**: Consulting with a psychiatrist or psychologist is crucial. They can guide you with proper diagnosis and treatment options.\n"
            "- **Maintain a Routine**: Regular schedules for sleep, meals, and activities can stabilize mood swings.\n"
            "- **Medication and Therapy**: Treatment often includes mood stabilizers and therapies such as Cognitive Behavioral Therapy (CBT).\n"
            "- **Build a Support System**: Talking to trusted friends, family, or support groups can make a big difference.\n"
            "- **Self-Care Practices**: Engage in calming activities like yoga, mindfulness, or journaling.\n\n"
            "Remember, you are not alone in this. Bipolar Disorder is treatable, and many people lead fulfilling lives with proper care. Take the first step toward reaching out for help—your mental health matters. Be happy! 😊"
        )
    elif prediction == "Personality disorder":
        return (
            "It seems like you're experiencing symptoms that may be related to a **Personality Disorder**. "
            "Personality disorders are a group of mental health conditions that can affect how a person thinks, feels, and behaves. "
            "It's important to remember that you are not alone, and with the right treatment and support, many people with personality disorders lead fulfilling lives.\n\n"
            "Here are some suggestions that may help:\n"
            "- **Consult a Mental Health Professional**: A psychiatrist or psychologist can help diagnose and provide therapy options tailored to your needs.\n"
            "- **Therapy**: Cognitive Behavioral Therapy (CBT) and Dialectical Behavior Therapy (DBT) are common therapies used to help manage symptoms of personality disorders.\n"
            "- **Medication**: In some cases, medication may be prescribed to help manage symptoms like anxiety or mood swings.\n"
            "- **Build Healthy Relationships**: It's important to establish supportive relationships with friends, family, and mental health professionals.\n"
            "- **Practice Self-Care**: Regular exercise, healthy eating, and mindfulness practices can help improve emotional regulation and overall well-being.\n\n"
            "Remember, seeking help is a brave first step, and with the right support, you can manage and improve your mental health. You are deserving of care and support. Be happy! 😊"
        )
    elif prediction == "Anxiety":
        return (
            "It seems like you're experiencing symptoms of **Anxiety**. Anxiety is a common mental health condition, and it's important to acknowledge your feelings. "
            "You're not alone, and with the right strategies, you can manage your symptoms and regain a sense of control.\n\n"
            "Here are some suggestions that may help:\n"
            "- **Practice Relaxation Techniques**: Deep breathing exercises, progressive muscle relaxation, and mindfulness meditation can help calm your mind and reduce physical symptoms.\n"
            "- **Stay Active**: Regular physical activity can reduce anxiety and boost your mood. Even a short walk or stretching exercises can help.\n"
            "- **Limit Caffeine and Sugar**: These can sometimes increase feelings of anxiety. Try to focus on a balanced diet and drink water regularly.\n"
            "- **Reach Out for Support**: Talking to a trusted friend, family member, or therapist can help you feel heard and less alone.\n"
            "- **Consider Professional Help**: If anxiety is significantly impacting your life, therapy (such as Cognitive Behavioral Therapy) or medication may be recommended by a healthcare provider.\n\n"
            "Remember, anxiety is manageable, and seeking support is a brave step. You're doing great, and it's okay to ask for help when needed. Be happy! 😊"
        )
    elif prediction == "NO":
        return (
            "It appears that you're not suffering from any significant mental health issues based on your input. "
            "However, it's always important to stay mindful of your well-being. If you start to notice symptoms like sadness, anxiety, or stress, don't hesitate to reach out for support.\n\n"
            "Remember, maintaining mental health is as important as physical health. Take care of yourself, and always feel free to seek support when needed!"
        )
    else:
        return "Sorry, I couldn't process your request. Please try again."

# Run the main function
if __name__ == '__main__':
    main()
