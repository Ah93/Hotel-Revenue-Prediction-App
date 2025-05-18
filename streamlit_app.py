import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model
model = joblib.load("catboost_model.pkl")

# Load your dataset (for visualizations)
data = pd.read_csv("hotel_booking.csv")  # Replace with actual CSV

# Pre-calculate visualizations data
data['total_nights'] = data['stays_in_weekend_nights'] + data['stays_in_week_nights']

cancellation_rate_by_hotel = (
    data.groupby('hotel')['is_canceled']
    .mean() * 100
).round(2)

avg_lead_time = data.groupby(['hotel', 'is_canceled'])['lead_time'].mean().unstack()

monthly_revenue = data.groupby(['hotel', 'arrival_date_month'])['adr'].sum().unstack().fillna(0)

avg_adr = data.groupby(['customer_type', 'hotel'])['adr'].mean().unstack()

parking_demand = data.groupby('hotel')['required_car_parking_spaces'].mean() * 100

special_requests_trend = data.groupby(['hotel', 'arrival_date_year'])['total_of_special_requests'].mean().unstack()

reservation_status_dist = data['reservation_status'].value_counts()

# Set up tabs
tab1, tab2 = st.tabs(["📈 Revenue Prediction", "📊 Analysis & Visualizations"])

# ========================
# 📈 Tab 1: Prediction
# ========================
with tab1:
    st.title("📈 Hotel Revenue Prediction App")
    st.write("Fill in the booking details below to estimate the expected revenue.")

    total_nights = st.number_input("Total Nights", min_value=0)
    stays_in_weekend_nights = st.number_input("Weekend Nights", min_value=0)
    stays_in_week_nights = st.number_input("Week Nights", min_value=0)
    customer_type_contract = st.checkbox("Customer Type: Contract", value=False)
    is_canceled = st.checkbox("Is Canceled", value=False)
    reservation_status_checkout = st.checkbox("Reservation Status: Check-Out", value=True)
    reservation_status_canceled = st.checkbox("Reservation Status: Canceled", value=False)
    lead_time = st.number_input("Lead Time (days before arrival)", min_value=0)
    adr = st.number_input("Average Daily Rate (ADR)", min_value=0.0)
    is_repeated_guest = st.checkbox("Is Repeated Guest", value=False)

    if st.button("Predict Revenue"):
        input_data = pd.DataFrame([{
            'total_nights': total_nights,
            'stays_in_weekend_nights': stays_in_weekend_nights,
            'stays_in_week_nights': stays_in_week_nights,
            'customer_type_Contract': int(customer_type_contract),
            'is_canceled': int(is_canceled),
            'reservation_status_Check-Out': int(reservation_status_checkout),
            'reservation_status_Canceled': int(reservation_status_canceled),
            'lead_time': lead_time,
            'adr': adr,
            'is_repeated_guest': int(is_repeated_guest)
        }])
        prediction = model.predict(input_data)[0]
        st.success(f"💰 Predicted Revenue: €{prediction:.2f}")

# ========================
# 📊 Tab 2: Visualizations
# ========================
with tab2:
    st.header("📊 Data Insights & Visualizations")

    # 1. Cancellation Rate by Hotel
    st.subheader("1. Cancellation Rate by Hotel")
    fig1, ax1 = plt.subplots()
    cancellation_rate_by_hotel.plot(kind='bar', color='skyblue', ax=ax1)
    ax1.set_title('Cancellation Rate by Hotel')
    ax1.set_ylabel('Cancellation Rate (%)')
    st.pyplot(fig1)

    # 2. Average Lead Time by Hotel and Cancellation Status
    st.subheader("2. Average Lead Time by Hotel and Cancellation Status")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    avg_lead_time.plot(kind='bar', ax=ax2)
    ax2.set_title('Average Lead Time')
    st.pyplot(fig2)

    # 3. Total Revenue by Hotel and Month
    st.subheader("3. Total Monthly Revenue by Hotel")
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    sns.heatmap(monthly_revenue, annot=True, fmt=".0f", cmap='YlGnBu', ax=ax3)
    ax3.set_title('Monthly Revenue')
    st.pyplot(fig3.figure)

    # 4. Average ADR by Customer Type and Hotel
    st.subheader("4. Average ADR by Customer Type and Hotel")
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    avg_adr.plot(kind='bar', ax=ax4)
    ax4.set_title('ADR by Customer Type & Hotel')
    st.pyplot(fig4)

    # 5. Parking Demand Rate by Hotel
    st.subheader("5. Parking Demand Rate by Hotel")
    fig5, ax5 = plt.subplots()
    parking_demand.plot(kind='bar', color='coral', ax=ax5)
    ax5.set_title('Parking Demand by Hotel')
    ax5.set_ylabel('Parking %')
    st.pyplot(fig5)

    # 6. Special Requests Trend by Year
    st.subheader("6. Special Requests by Hotel Over Years")
    fig6, ax6 = plt.subplots(figsize=(8, 5))
    special_requests_trend.T.plot(kind='line', marker='o', ax=ax6)
    ax6.set_title('Special Requests Trend')
    st.pyplot(fig6)

    # 7. Reservation Status Distribution
    st.subheader("7. Reservation Status Distribution")
    fig7, ax7 = plt.subplots()
    reservation_status_dist.plot(kind='pie', autopct='%1.1f%%', startangle=90,
                                 colors=sns.color_palette('pastel'), ax=ax7)
    ax7.set_ylabel('')
    ax7.set_title('Reservation Status Share')
    st.pyplot(fig7)
