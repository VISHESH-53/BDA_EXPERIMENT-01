import streamlit as st
import pandas as pd
import plotly.express as px
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, udf, concat, lit
)
from pyspark.sql.types import StringType
from textblob import TextBlob
import time

# Page configuration
st.set_page_config(
    page_title="Real-Time Sentiment Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("Real-Time Twitter/X Sentiment Dashboard")
st.write("Sentiment analysis using Apache Spark Structured Streaming")

# Sample social media posts
posts = [
    "Artificial intelligence is amazing!",
    "I love this new technology.",
    "This product is excellent.",
    "The service is very disappointing.",
    "I hate these problems.",
    "This is the worst experience.",
    "The weather is normal today.",
    "The meeting is scheduled for tomorrow.",
    "AI will transform the world.",
    "This update is terrible."
]
import os
import shutil

java_path = shutil.which("java")

if java_path:
    print("Java executable:", java_path)
else:
    raise RuntimeError("Java is not installed or not on PATH")
# Create Spark session
@st.cache_resource
def create_spark():
    return (
        SparkSession.builder
        .appName("TwitterSentimentDashboard")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )

spark = create_spark()
spark.sparkContext.setLogLevel("ERROR")

# Sentiment classification function
def analyze_sentiment(text):
    score = TextBlob(text).sentiment.polarity

    if score > 0.1:
        return "Positive"
    elif score < -0.1:
        return "Negative"
    else:
        return "Neutral"

sentiment_udf = udf(analyze_sentiment, StringType())

# Start simulated streaming only once
@st.cache_resource
def start_stream():
    source = (
        spark.readStream
        .format("rate")
        .option("rowsPerSecond", 1)
        .load()
    )

    # Generate sample posts from streaming rows
    stream = source.withColumn(
        "post",
        when(
            (col("value") % 10) == 0, lit(posts[0])
        ).when(
            (col("value") % 10) == 1, lit(posts[1])
        ).when(
            (col("value") % 10) == 2, lit(posts[2])
        ).when(
            (col("value") % 10) == 3, lit(posts[3])
        ).when(
            (col("value") % 10) == 4, lit(posts[4])
        ).when(
            (col("value") % 10) == 5, lit(posts[5])
        ).when(
            (col("value") % 10) == 6, lit(posts[6])
        ).when(
            (col("value") % 10) == 7, lit(posts[7])
        ).when(
            (col("value") % 10) == 8, lit(posts[8])
        ).otherwise(lit(posts[9]))
    )

    stream = stream.withColumn(
        "sentiment", sentiment_udf(col("post"))
    )

    query = (
        stream.writeStream
        .format("memory")
        .queryName("sentiment_stream")
        .outputMode("append")
        .option("checkpointLocation", "spark_checkpoint")
        .start()
    )

    return query

query = start_stream()

# Dashboard refresh
time.sleep(2)

try:
    df = spark.sql(
        "SELECT timestamp, post, sentiment "
        "FROM sentiment_stream ORDER BY timestamp DESC"
    ).toPandas()
except Exception:
    df = pd.DataFrame(
        columns=["timestamp", "post", "sentiment"]
    )

# Display metrics
total = len(df)
positive = int((df["sentiment"] == "Positive").sum())
negative = int((df["sentiment"] == "Negative").sum())
neutral = int((df["sentiment"] == "Neutral").sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Posts", total)
c2.metric("Positive", positive)
c3.metric("Negative", negative)
c4.metric("Neutral", neutral)

# Sentiment distribution chart
st.subheader("Sentiment Distribution")

counts = pd.DataFrame({
    "Sentiment": ["Positive", "Negative", "Neutral"],
    "Count": [positive, negative, neutral]
})

fig = px.pie(
    counts,
    names="Sentiment",
    values="Count",
    title="Distribution of Post Sentiment",
    hole=0.35
)

st.plotly_chart(fig, use_container_width=True)

# Recent posts
st.subheader("Recent Social Media Posts")

if not df.empty:
    st.dataframe(
        df[["timestamp", "post", "sentiment"]],
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("Waiting for streaming data...")

st.caption(
    "Demonstration uses simulated posts. "
    "Sentiment labels are generated using TextBlob polarity."
)

# Refresh dashboard periodically
st.button(
    "Refresh Dashboard",
    on_click=lambda: st.rerun()
)
