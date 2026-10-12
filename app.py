
import os
import shutil
import time
import random
from pathlib import Path
from datetime import datetime, timezone

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob

from pyspark.sql import SparkSession
from pyspark.sql.functions import col


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="SocialPulse | Sentiment Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0b1120;
        color: #f1f5f9;
    }

    [data-testid="stSidebar"] {
        background-color: #111827;
    }

    [data-testid="stMetric"] {
        background-color: #172033;
        border: 1px solid #293449;
        padding: 18px;
        border-radius: 12px;
    }

    div[data-testid="stMetricValue"] {
        color: #60a5fa;
    }

    .main-title {
        font-size: 38px;
        font-weight: 800;
        color: #60a5fa;
        margin-bottom: 0;
    }

    .subtitle {
        color: #94a3b8;
        font-size: 15px;
        margin-top: 4px;
    }

    .section-title {
        font-size: 22px;
        font-weight: 700;
        color: #e2e8f0;
        margin-top: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# JAVA CONFIGURATION FOR STREAMLIT CLOUD
# --------------------------------------------------

def configure_java():
    """Find Java and configure JAVA_HOME when available."""

    java_path = shutil.which("java")

    if java_path:
        resolved_java = Path(java_path).resolve()
        java_home = resolved_java.parent.parent

        os.environ["JAVA_HOME"] = str(java_home)
        return True, java_path

    possible_paths = [
        "/usr/lib/jvm/java-21-openjdk-amd64/bin/java",
        "/usr/lib/jvm/java-17-openjdk-amd64/bin/java",
        "/usr/lib/jvm/default-java/bin/java",
    ]

    for candidate in possible_paths:
        if Path(candidate).exists():
            java_path = candidate
            os.environ["JAVA_HOME"] = str(
                Path(candidate).resolve().parent.parent
            )
            os.environ["PATH"] = (
                str(Path(candidate).parent)
                + os.pathsep
                + os.environ.get("PATH", "")
            )
            return True, java_path

    return False, None


# --------------------------------------------------
# SAMPLE SOCIAL MEDIA POSTS
# --------------------------------------------------

SAMPLE_POSTS = [
    ("The new technology update is absolutely amazing!", "Technology"),
    ("This product is disappointing and full of bugs.", "Technology"),
    ("The latest AI tools are improving productivity.", "AI"),
    ("I am frustrated with the latest software update.", "Technology"),
    ("What an excellent performance by the team!", "Sports"),
    ("The service was terrible and very slow.", "Business"),
    ("This innovation could change the future.", "AI"),
    ("The market outlook looks promising today.", "Finance"),
    ("Another disappointing result for the company.", "Finance"),
    ("I love how easy this application is to use.", "Technology"),
    ("The new policy has received mixed reactions.", "Politics"),
    ("This is a great step toward a better future.", "Politics"),
    ("The customer support was unhelpful.", "Business"),
    ("A brilliant achievement by the research team.", "AI"),
    ("The experience was average, nothing special.", "Business"),
    ("Investors are concerned about market volatility.", "Finance"),
    ("The match was exciting from start to finish!", "Sports"),
    ("The team played poorly and wasted opportunities.", "Sports"),
    ("AI research continues to produce impressive results.", "AI"),
    ("The new service is okay, but needs improvement.", "Business"),
]


def classify_sentiment(text):
    """Classify text using TextBlob polarity."""

    polarity = TextBlob(str(text)).sentiment.polarity

    if polarity > 0.05:
        return "Positive"
    elif polarity < -0.05:
        return "Negative"
    return "Neutral"


def create_demo_posts(count=50):
    """Generate demo posts for the dashboard."""

    rows = []
    now = datetime.now(timezone.utc)

    for i in range(count):
        text, topic = random.choice(SAMPLE_POSTS)
        sentiment = classify_sentiment(text)

        rows.append(
            {
                "post_id": i + 1,
                "timestamp": now,
                "text": text,
                "topic": topic,
                "sentiment": sentiment,
                "polarity": TextBlob(text).sentiment.polarity,
                "source": "Simulated",
            }
        )

    return pd.DataFrame(rows)


# --------------------------------------------------
# SPARK STREAMING
# --------------------------------------------------

@st.cache_resource(show_spinner="Starting Apache Spark...")
def start_spark_stream():
    """
    Start a cached Spark Structured Streaming query.
    The rate source generates one row per second.
    """

    available, java_path = configure_java()

    if not available:
        raise RuntimeError(
            "Java was not found. Install Java using packages.txt "
            "and verify the Streamlit Cloud logs."
        )

    spark = (
        SparkSession.builder
        .appName("TwitterSentimentDashboard")
        .master("local[2]")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.driver.memory", "512m")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    # Avoid duplicate query names in the same Spark session.
    query_name = "socialpulse_stream"

    if spark.streams.active:
        for active_query in spark.streams.active:
            if active_query.name == query_name:
                return spark, active_query

    stream_df = (
        spark.readStream
        .format("rate")
        .option("rowsPerSecond", 1)
        .option("numPartitions", 1)
        .load()
    )

    query = (
        stream_df
        .writeStream
        .format("memory")
        .queryName(query_name)
        .outputMode("append")
        .trigger(processingTime="2 seconds")
        .start()
    )

    return spark, query


def get_spark_events():
    """Read events currently available in the Spark memory sink."""

    spark, query = start_spark_stream()

    # Give the first micro-batch a short opportunity to arrive.
    if query.isActive:
        time.sleep(1.0)

    events = spark.sql(
        "SELECT timestamp, value FROM socialpulse_stream"
    ).toPandas()

    if events.empty:
        return pd.DataFrame()

    # Map Spark stream events to sample social-media messages.
    posts = []

    for _, event in events.iterrows():
        index = int(event["value"]) % len(SAMPLE_POSTS)
        text, topic = SAMPLE_POSTS[index]
        sentiment = classify_sentiment(text)

        posts.append(
            {
                "post_id": int(event["value"]),
                "timestamp": event["timestamp"],
                "text": text,
                "topic": topic,
                "sentiment": sentiment,
                "polarity": TextBlob(text).sentiment.polarity,
                "source": "Spark Stream",
            }
        )

    return pd.DataFrame(posts)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<p class="main-title">📊 SocialPulse</p>',
    unsafe_allow_html=True,
)

st.markdown(
    '<p class="subtitle">'
    'Real-Time Social Media Sentiment Analytics '
    '| Apache Spark Structured Streaming'
    '</p>',
    unsafe_allow_html=True,
)

st.divider()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("⚙️ Dashboard Controls")

st.sidebar.caption("Configure your analytics view.")

topic_options = [
    "All Topics",
    "AI",
    "Technology",
    "Finance",
    "Sports",
    "Business",
    "Politics",
]

selected_topic = st.sidebar.selectbox(
    "Select topic",
    topic_options,
)

sentiment_options = [
    "All Sentiments",
    "Positive",
    "Negative",
    "Neutral",
]

selected_sentiment = st.sidebar.selectbox(
    "Filter sentiment",
    sentiment_options,
)

data_mode = st.sidebar.radio(
    "Data source",
    ["Spark Streaming", "Demo Data"],
    help=(
        "Spark Streaming reads generated events from Spark's rate source. "
        "Demo Data creates a sample dataset directly in Python."
    ),
)

refresh = st.sidebar.button(
    "🔄 Refresh Dashboard",
    use_container_width=True,
)

st.sidebar.divider()

st.sidebar.markdown("### About")
st.sidebar.caption(
    "Demonstration project using Spark Structured Streaming, "
    "TextBlob, Pandas, Plotly, and Streamlit."
)

st.sidebar.warning(
    "This demo does not connect to the live X/Twitter API. "
    "Posts are simulated."
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

posts_df = pd.DataFrame()
spark_status = "Demo mode"

if data_mode == "Spark Streaming":
    try:
        with st.spinner("Connecting to Spark Structured Streaming..."):
            posts_df = get_spark_events()

        if not posts_df.empty:
            spark_status = "Spark stream connected"
        else:
            st.info(
                "Spark is running, but no stream events are available yet. "
                "Refresh the dashboard in a moment."
            )

    except Exception as exc:
        st.error(
            "Spark Streaming could not start. Check the Java installation "
            "and Streamlit Cloud logs. You can switch to Demo Data "
            "to view the dashboard."
        )
        with st.expander("Technical details"):
            st.code(str(exc))

        posts_df = pd.DataFrame()

else:
    posts_df = create_demo_posts(50)
    spark_status = "Demo data generated"


if not posts_df.empty:
    posts_df["timestamp"] = pd.to_datetime(
        posts_df["timestamp"], errors="coerce"
    )

    filtered_df = posts_df.copy()

    if selected_topic != "All Topics":
        filtered_df = filtered_df[
            filtered_df["topic"] == selected_topic
        ]

    if selected_sentiment != "All Sentiments":
        filtered_df = filtered_df[
            filtered_df["sentiment"] == selected_sentiment
        ]

else:
    filtered_df = pd.DataFrame()


# --------------------------------------------------
# STATUS
# --------------------------------------------------

status_col1, status_col2 = st.columns([2, 1])

with status_col1:
    st.caption(f"**Pipeline status:** {spark_status}")

with status_col2:
    st.caption(
        "Last dashboard load: "
        + datetime.now().strftime("%d %b %Y, %H:%M:%S")
    )


# --------------------------------------------------
# KPI METRICS
# --------------------------------------------------

st.markdown(
    '<p class="section-title">📌 Key Performance Indicators</p>',
    unsafe_allow_html=True,
)

if not filtered_df.empty:
    total_posts = len(filtered_df)

    positive_count = int(
        (filtered_df["sentiment"] == "Positive").sum()
    )
    negative_count = int(
        (filtered_df["sentiment"] == "Negative").sum()
    )
    neutral_count = int(
        (filtered_df["sentiment"] == "Neutral").sum()
    )

    positive_percentage = (
        positive_count / total_posts * 100 if total_posts else 0
    )

    avg_polarity = float(filtered_df["polarity"].mean())

else:
    total_posts = 0
    positive_count = 0
    negative_count = 0
    neutral_count = 0
    positive_percentage = 0
    avg_polarity = 0


metric1, metric2, metric3, metric4 = st.columns(4)

metric1.metric(
    "Total Posts",
    f"{total_posts:,}",
)

metric2.metric(
    "Positive Posts",
    f"{positive_count:,}",
)

metric3.metric(
    "Negative Posts",
    f"{negative_count:,}",
)

metric4.metric(
    "Positive Rate",
    f"{positive_percentage:.1f}%",
)

st.divider()


# --------------------------------------------------
# CHARTS
# --------------------------------------------------

st.markdown(
    '<p class="section-title">📈 Sentiment Analytics</p>',
    unsafe_allow_html=True,
)

if not filtered_df.empty:
    chart_col1, chart_col2 = st.columns(2)

    sentiment_counts = (
        filtered_df["sentiment"]
        .value_counts()
        .reindex(["Positive", "Negative", "Neutral"], fill_value=0)
        .rename_axis("Sentiment")
        .reset_index(name="Posts")
    )

    with chart_col1:
        st.subheader("Sentiment Distribution")

        pie_fig = px.pie(
            sentiment_counts,
            names="Sentiment",
            values="Posts",
            hole=0.55,
            title="Positive vs Negative vs Neutral",
        )

        pie_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend_title_text="Sentiment",
        )

        st.plotly_chart(pie_fig, use_container_width=True)

    with chart_col2:
        st.subheader("Posts by Topic")

        topic_counts = (
            filtered_df["topic"]
            .value_counts()
            .rename_axis("Topic")
            .reset_index(name="Posts")
        )

        bar_fig = px.bar(
            topic_counts,
            x="Topic",
            y="Posts",
            color="Topic",
            title="Topic-wise Post Volume",
            text="Posts",
        )

        bar_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )

        st.plotly_chart(bar_fig, use_container_width=True)

    # Sentiment polarity chart
    st.subheader("Sentiment Polarity")

    polarity_fig = px.histogram(
        filtered_df,
        x="polarity",
        nbins=15,
        title="Distribution of TextBlob Polarity",
    )

    polarity_fig.add_vline(
        x=0,
        line_dash="dash",
        annotation_text="Neutral boundary",
    )

    polarity_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Polarity (-1 to +1)",
        yaxis_title="Number of Posts",
    )

    st.plotly_chart(polarity_fig, use_container_width=True)

else:
    st.info(
        "No posts match the selected filters. "
        "Try another topic or sentiment, or refresh the dashboard."
    )


# --------------------------------------------------
# RECENT POSTS
# --------------------------------------------------

st.divider()

st.markdown(
    '<p class="section-title">🗞️ Recent Social Media Posts</p>',
    unsafe_allow_html=True,
)

if not filtered_df.empty:
    recent_df = filtered_df.sort_values(
        "timestamp", ascending=False
    ).head(20)

    display_df = recent_df[
        [
            "timestamp",
            "text",
            "topic",
            "sentiment",
            "polarity",
            "source",
        ]
    ].copy()

    display_df["timestamp"] = display_df["timestamp"].dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    display_df["polarity"] = display_df["polarity"].round(3)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "timestamp": st.column_config.TextColumn("Timestamp"),
            "text": st.column_config.TextColumn(
                "Post Text", width="large"
            ),
            "topic": st.column_config.TextColumn("Topic"),
            "sentiment": st.column_config.TextColumn("Sentiment"),
            "polarity": st.column_config.NumberColumn(
                "Polarity", format="%.3f"
            ),
            "source": st.column_config.TextColumn("Source"),
        },
    )

    csv_data = filtered_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Posts as CSV",
        data=csv_data,
        file_name="socialpulse_sentiment_data.csv",
        mime="text/csv",
    )

else:
    st.write("No posts available to display.")


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "SocialPulse | Academic Big Data Analytics Project | "
    "Apache Spark Structured Streaming + Streamlit + Plotly"
)

st.caption(
    "Note: This prototype uses simulated posts. "
    "Live X/Twitter analysis requires an authorized data source and API access."
)
