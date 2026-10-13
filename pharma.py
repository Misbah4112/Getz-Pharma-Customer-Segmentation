# K-Means discovers groups without being given group labels.
# This is called unsupervised learning.

import plotly.express as px
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


st.set_page_config(
    page_title="Getz Pharma Customer Segmentation",
    page_icon="💊",
    layout="wide"
)

st.title("💊 Getz Pharma Customer Segmentation")
st.subheader("Unsupervised Learning with K-Means")

st.caption(
    "Explore customer buying behaviour and discover customer groups."
)

st.info(
    "This app groups customers based on how they buy pharmaceutical "
    "products. These groups can help you decide which customers "
    "to contact and what to offer them."
)

with st.popover("What is K-Means?"):
    st.write(
        "K-Means groups customers with similar characteristics. "
        "K is the number of groups we ask it to create."
    )


# Load the dataset
DATA_FILE = "getz_pharma_customer_segmentation_1000.csv"
df = pd.read_csv(DATA_FILE)


# Overview of the full dataset
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("Customers", len(df))

with m2:
    st.metric("Cities", df["City"].nunique())

with m3:
    st.metric(
        "Avg Monthly Orders",
        f"{df['Monthly_Orders'].mean():.1f}"
    )

with m4:
    st.metric(
        "Avg Monthly Spend",
        f"PKR {df['Monthly_Spend_PKR'].mean():,.0f}"
    )


# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 1. Dataset",
        "📈 2. Choose K",
        "🤖 3. Customer Segments",
        "🧠 4. What We Learned"
    ]
)


# Sidebar controls
with st.sidebar:
    st.header("💊 Dataset Filters")
    st.caption("These filters affect the table and charts in Tab 1.")

    with st.form("dataset_filters"):
        selected_cities = st.multiselect(
            "Choose Cities",
            options=sorted(df["City"].dropna().unique())
        )

        selected_types = st.multiselect(
            "Choose Customer Types",
            options=sorted(df["Customer_Type"].dropna().unique())
        )

        st.form_submit_button("Apply Filters")


# Apply the submitted filters
filtered_df = df.copy()

if selected_cities:
    filtered_df = filtered_df[
        filtered_df["City"].isin(selected_cities)
    ]

if selected_types:
    filtered_df = filtered_df[
        filtered_df["Customer_Type"].isin(selected_types)
    ]


# TAB 1: Dataset
with tab1:
    st.header("Step 1: Understand the Data")

    st.write(
        "Before using Machine Learning, we first understand "
        "what each column means."
    )

    st.write(f"Matching customers: {len(filtered_df)}")
    st.caption("Showing the first 20 matching customers.")

    st.dataframe(
        filtered_df.head(20),
        width="stretch",
        hide_index=True
    )

    with st.expander("What does each column mean?"):
        st.write("**Customer_ID:** Unique code for each customer.")
        st.write("**Outlet_Name:** Name of the pharmacy or business.")
        st.write("**City:** City where the customer is located.")
        st.write(
            "**Customer_Type:** Retail pharmacy, hospital pharmacy, "
            "or distributor."
        )
        st.write("**Monthly_Orders:** Orders placed per month.")
        st.write(
            "**Avg_Order_Value_PKR:** Average amount spent per order."
        )
        st.write("**Products_Purchased:** Number of products purchased.")
        st.write(
            "**Payment_Delay_Days:** Number of days payment is delayed."
        )
        st.write(
            "**Return_Rate_Percent:** Percentage of purchases returned."
        )
        st.write(
            "**Months_With_Company:** Months the customer has been "
            "with the company."
        )
        st.write(
            "**Days_Since_Last_Order:** Days since the latest order."
        )
        st.write(
            "**Monthly_Spend_PKR:** Total monthly spending in rupees."
        )

    st.subheader("Missing Values Check")
    st.caption("This check covers the full dataset.")

    missing_values = df.isna().sum()

    st.dataframe(
        missing_values.reset_index().rename(
            columns={"index": "Column", 0: "Missing Values"}
        ),
        width="stretch",
        hide_index=True
    )

    st.subheader("Explore the Dataset with Different Charts")

    chart_type = st.selectbox(
        "Choose a Chart",
        ["Histogram", "Box Plot", "Scatter Plot"]
    )

    numeric_columns = filtered_df.select_dtypes(
        include="number"
    ).columns.tolist()

    # Exclude identifiers from chart measurements
    numeric_columns = [
        col for col in numeric_columns
        if col not in ["Customer_ID", "Cluster"]
    ]

    if filtered_df.empty:
        st.info("No customers match your filters.")

    elif not numeric_columns:
        st.info("No numeric columns are available.")

    else:
        x_column = st.selectbox(
            "Choose a Numeric Column",
            options=numeric_columns,
            key="dataset_x"
        )

        if chart_type == "Histogram":
            fig = px.histogram(
                filtered_df,
                x=x_column,
                nbins=20,
                title=f"Distribution of {x_column}"
            )

        elif chart_type == "Box Plot":
            fig = px.box(
                filtered_df,
                x="Customer_Type",
                y=x_column,
                color="Customer_Type",
                title=f"{x_column} by Customer Type"
            )

        else:
            y_column = st.selectbox(
                "Choose the Y-axis Column",
                options=numeric_columns,
                key="dataset_y"
            )

            fig = px.scatter(
                filtered_df,
                x=x_column,
                y=y_column,
                color="Customer_Type",
                hover_name="Outlet_Name",
                title=f"{x_column} vs {y_column}"
            )

        # Display whichever chart was selected
        st.plotly_chart(fig, width="stretch")

    with open(DATA_FILE, "rb") as file:
        st.download_button(
            label="Download Original Dataset",
            data=file,
            file_name=DATA_FILE,
            mime="text/csv"
        )


# Select features for clustering
features = [
    "Monthly_Orders",
    "Avg_Order_Value_PKR",
    "Products_Purchased",
    "Payment_Delay_Days",
    "Days_Since_Last_Order"
]

X = df[features]

# Put features on comparable scales
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# TAB 2: Choose K
with tab2:
    st.header("Step 2: How Many Groups Should We Make?")
    st.subheader("Elbow Method")

    st.write(
        "We train K-Means with different values of K and compare "
        "their inertia. Inertia measures how close customers are "
        "to their cluster centres. We look for a bend where adding "
        "more groups gives a smaller improvement."
    )

    k_values = list(range(1, 8))
    inertia_values = []

    for k in k_values:
        elbow_model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        elbow_model.fit(X_scaled)
        inertia_values.append(elbow_model.inertia_)

    # Build and display the chart after all models finish
    elbow_df = pd.DataFrame(
        {
            "K": k_values,
            "Inertia": inertia_values
        }
    )

    st.line_chart(
        elbow_df,
        x="K",
        y="Inertia",
        x_label="Number of Clusters (K)",
        y_label="Inertia"
    )

    st.info(
        "Start with K = 3 for this classroom example. "
        "Check the chart to see whether it supports this choice."
    )

    selected_k = st.slider(
        "Choose the Number of Clusters (K)",
        min_value=2,
        max_value=6,
        value=3,
        step=1
    )


# Train the final model using the selected K
model = KMeans(
    n_clusters=selected_k,
    random_state=42,
    n_init=10
)

cluster_labels = model.fit_predict(X_scaled)

df_clustered = df.copy()
df_clustered["Cluster"] = cluster_labels


# Calculate a summary to help assign segment names
cluster_summary = (
    df_clustered.groupby("Cluster")
    .agg(
        Customers=("Customer_ID", "count"),
        Avg_Monthly_Orders=("Monthly_Orders", "mean"),
        Avg_Order_Value_PKR=("Avg_Order_Value_PKR", "mean"),
        Avg_Products_Purchased=("Products_Purchased", "mean"),
        Avg_Payment_Delay_Days=("Payment_Delay_Days", "mean"),
        Avg_Days_Since_Last_Order=("Days_Since_Last_Order", "mean"),
        Avg_Monthly_Spend_PKR=("Monthly_Spend_PKR", "mean")
    )
    .round(1)
)


# Assign names after clustering
if selected_k == 3:
    ordered_clusters = (
        cluster_summary["Avg_Monthly_Spend_PKR"]
        .sort_values()
        .index
        .tolist()
    )

    segment_names = {
        ordered_clusters[0]: "Low Engagement",
        ordered_clusters[1]: "Regular Customers",
        ordered_clusters[2]: "High Value Loyal Customers"
    }

    df_clustered["Segment"] = (
        df_clustered["Cluster"].map(segment_names)
    )

else:
    df_clustered["Segment"] = (
        "Cluster " + df_clustered["Cluster"].astype(str)
    )


# TAB 3: Customer segments
with tab3:
    st.header("Step 3: Explore the Groups K-Means Discovered")
    st.subheader("How Many Customers Are in Each Group?")

    segment_counts = (
        df_clustered["Segment"]
        .value_counts()
        .rename_axis("Segment")
        .reset_index(name="Customers")
    )

    show_percent = st.toggle("Show Percentages")
    chart_data = segment_counts.copy()

    if show_percent:
        chart_data["Percentage"] = (
            chart_data["Customers"]
            / chart_data["Customers"].sum()
            * 100
        )

        st.bar_chart(
            chart_data,
            x="Segment",
            y="Percentage",
            x_label="Customer Segments",
            y_label="Customers (%)"
        )

    else:
        st.bar_chart(
            chart_data,
            x="Segment",
            y="Customers",
            x_label="Customer Segments",
            y_label="Number of Customers"
        )

    # Outside the toggle branches, but inside Tab 3
    st.subheader("Visualize the Clusters")

    st.scatter_chart(
        df_clustered,
        x="Monthly_Orders",
        y="Avg_Order_Value_PKR",
        color="Segment",
        size="Products_Purchased",
        x_label="Monthly Orders",
        y_label="Average Order Value (PKR)"
    )

    st.subheader("Cluster Summary")

    segment_summary = (
        df_clustered.groupby("Segment")
        .agg(
            Customers=("Customer_ID", "count"),
            Avg_Monthly_Orders=("Monthly_Orders", "mean"),
            Avg_Order_Value_PKR=("Avg_Order_Value_PKR", "mean"),
            Avg_Products_Purchased=("Products_Purchased", "mean"),
            Avg_Payment_Delay_Days=("Payment_Delay_Days", "mean"),
            Avg_Days_Since_Last_Order=("Days_Since_Last_Order", "mean"),
            Avg_Monthly_Spend_PKR=("Monthly_Spend_PKR", "mean")
        )
        .round(1)
        .reset_index()
    )

    st.dataframe(
        segment_summary,
        width="stretch",
        hide_index=True
    )

    if selected_k == 3:
        st.success(
            "The model created three groups. We assigned names "
            "by ranking their average monthly spending."
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            st.subheader("High Value Loyal Customers")
            st.write(
                "The group with the highest average monthly spending."
            )

        with c2:
            st.subheader("Regular Customers")
            st.write(
                "The group with the middle average monthly spending."
            )

        with c3:
            st.subheader("Low Engagement")
            st.write(
                "The group with the lowest average monthly spending."
            )

        st.caption(
            "These are descriptive labels. Spending alone does not "
            "prove loyalty or engagement; check the other averages too."
        )

    # Available for every selected K
    st.subheader("Explore Individual Customers")

    segment_filter = st.selectbox(
        "Choose a Segment",
        options=sorted(df_clustered["Segment"].unique())
    )

    filtered = df_clustered[
        df_clustered["Segment"] == segment_filter
    ]

    st.dataframe(
        filtered,
        width="stretch",
        hide_index=True
    )

    st.subheader("Customer Segments Across Cities")

    st.write(
        "This table shows how many customers from each segment "
        "belong to each city."
    )

    city_segments = pd.crosstab(
        df_clustered["City"],
        df_clustered["Segment"]
    )

    st.dataframe(city_segments, width="stretch")


# TAB 4: What we learned
with tab4:
    st.header("Step 4: What We Learned")

    st.subheader("1. Understanding Our Customers")
    st.write(
        "We studied monthly orders, average order value, products "
        "purchased, payment delays, and days since the last order."
    )

    st.subheader("2. Making the Numbers Comparable")
    st.write(
        "Spending amounts are much larger numbers than order counts. "
        "StandardScaler puts the features on comparable scales "
        "so large numbers do not dominate the grouping."
    )

    st.subheader("3. Discovering Customer Groups")
    st.write(
        "K-Means groups customers with similar behaviour. "
        "We did not give it group labels beforehand. "
        "This is called unsupervised learning."
    )

    st.metric("Number of Groups Selected", selected_k)

    st.subheader("4. Choosing the Number of Groups")
    st.write(
        "The elbow chart compares different numbers of groups. "
        "We look for a bend where adding more groups gives "
        "a smaller improvement."
    )

    st.subheader("5. Understanding Each Group")
    st.write(
        "The summary table compares average orders, spending, "
        "payment delays, and recent activity. We assign group "
        "names after examining the results."
    )

    st.subheader("6. How the Company Can Use These Groups")
    st.write(
        "• Higher-spending customers: consider suitable offers "
        "and personalised support.\n\n"
        "• Regular customers: suggest relevant products "
        "and encourage repeat orders.\n\n"
        "• Less active customers: follow up and understand "
        "why they are ordering less."
    )

    st.info(
        "High spending alone does not prove customer loyalty. "
        "Check order frequency, recent activity, and payment "
        "behaviour before deciding how to approach a customer."
    )

    st.subheader("7. Reading Our Visuals")
    st.write(
        "• Histogram: shows how numerical values are distributed.\n\n"
        "• Box plot: compares the spread of values across customer types.\n\n"
        "• Bar chart: shows customer counts or percentages "
        "for each segment.\n\n"
        "• Scatter chart: compares two numerical measurements.\n\n"
        "• City–segment table: shows customer counts "
        "for each segment in each city."
    )