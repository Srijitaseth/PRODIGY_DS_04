import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

column_names = ['ID', 'entity', 'sentiment', 'text']

file_path = '/Users/srijitaseth/Downloads/archive/twitter_training.csv' 
data = pd.read_csv(file_path, names=column_names)

print(data.head())

print("Columns in the dataset:", data.columns)
print(data.isnull().sum())
data.dropna(inplace=True)
print(data.describe())
sentiment_counts = data['sentiment'].value_counts()
print(sentiment_counts)

plt.figure(figsize=(10, 6))
sns.countplot(x='sentiment', data=data, palette='viridis')
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

entity_counts = data['entity'].value_counts().head(10)
print(entity_counts)
plt.figure(figsize=(12, 8))
sns.barplot(x=entity_counts.values, y=entity_counts.index, palette='viridis')
plt.title('Top 10 Entities Mentioned in Tweets')
plt.xlabel('Count')
plt.ylabel('Entity')
plt.show()

top_entities = entity_counts.index
top_entity_data = data[data['entity'].isin(top_entities)]

plt.figure(figsize=(14, 10))
sns.countplot(x='entity', hue='sentiment', data=top_entity_data, palette='viridis')
plt.title('Sentiment Distribution for Top 10 Entities')
plt.xlabel('Entity')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()
