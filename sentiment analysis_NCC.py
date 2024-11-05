#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('stopwords')
nltk.download('vader_lexicon')

data = pd.read_excel('output.Norfolk.xlsx')

stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)

exclude_words = {'people', 'ncc', 'always', 'nan', 'one', 'time', 'would', 'think', 'make', 'need', 'way', 'much', 'sure', 'work', 'person', 'times', 'tries',
                 'meaning', 'like', 'council', 'service', 'others', 'feel', 'many', 'sometimes', 'staff', 'little', 'working','things', 'well', 'hard',
                 'good', 'organisation', 'answer', 'someone', 'culture','questions','get','services','help','listen','needs','lack',
                'help','know','trying','bit','say','open','could','money','change','question','manager','decision','different','whole','rather',
                'management','decisions','managers','still','big','less','understand','word','lot','within','issues','often',
                'large','right','services','get','ways','thing','forward','norfolk','ca','ideas','back','wants','changes','resources'
                ,'better','sorry','pay','go','find','us','keep','says','getting','though','yet','made','take','ground','place','going',
                'everything','years','another','done','day','face','role','intentioned','really','every','everyone'} 

word_categories = {
    'supportive': 'Positive',
    'reliable': 'Positive',
    'caring': 'Positive',
    'confused': 'Negative',
    'friendly': 'Positive',
    'complex': 'Neutral',
    'dependable': 'Positive',
    'steady': 'Positive',
    'question': 'Neutral',
    'compassionate': 'Positive',
    'competent': 'Positive',
    'complicated': 'Negative',
    'inconsistent': 'Negative',
    'trustworthy': 'Positive',
    'fair': 'Positive',
    'conflicted': 'Negative',
    'understanding': 'Positive',
    'frustrating': 'Negative',
    'best': 'Positive',
    'bureaucratic': 'Negative',
    'chaotic': 'Negative',
    'right': 'Positive',
    'disorganised': 'Negative',
    'difficult': 'Negative',
    'trying': 'Neutral',
    'services': 'Neutral',
    'changes': 'Neutral',
    'nice': 'Positive',
    'better': 'Positive',
    'individual': 'Neutral',
    'average': 'Neutral',
    'anxious': 'Negative',
    'disaster': 'Negative',
    'active': 'Neutral',
    'awkward': 'Negative',
    'unreliable': 'Negative',
    'authoritarian': 'Negative',
    'insensitive': 'Negative',
    'ambivolent': 'Negative',
    'aged': 'Negative',
    'annoying': 'Negative',
    'accommodating': 'Positive',
    'apathetic': 'Positive',
    'a swan': 'Neutral',
    'adaptable': 'Positive',
    'amicable': 'Positive',
    'abusive': 'Negative',
    'accepting': 'Positive',
    'amiable': 'Positive',
    'anonymous': 'Neutral',
    'absent': 'Negative',
    'aloof': 'Negative',
    'altruistic': 'Positive',
    'deceitful': 'Negative',
    'well-rounded': 'Positive',
    'authoritive': 'Positive',
    'assessor': 'Neutral',
    'arrogant': 'Negative',
    'athletic': 'Positive',
    'personable': 'Positive',
    'bully': 'Negative',
    'misogynistic': 'Negative',
    'agenda-driven': 'Neutral',
    'aspiring': 'Positive',
    'assertive': 'Neutral',
    'assiduous': 'Positive',
    'allrounder': 'Neutral',
    'accountability': 'Positive',
    'acquiescent': 'Neutral',
    'archaic': 'Negative',
    'apocolyptic': 'Negative',
    'trier': 'Positive',
    'autocratic': 'Negative',
    'leader': 'Positive',
    'exciting': 'Positive',
    'comforting': 'Positive',
    'nepotistic': 'Negative',
    'underachieving': 'Negative',
    'listens': 'Positive',
    'laughable': 'Negative',
    'artistic': 'Positive',
    'clumsy': 'Negative',
    'incompetent': 'Negative',
    'mixed up': 'Negative',
    'bland':'Neutral',
    'grey':'Neutral',
    'overwhelmed':'Negative',
    'challenging':'Neutral',
    'ok':'Neutral',
    'huge':'Neutral',
    'controlling':'Negative',
    'approachable':'Positive',
    'hardworking':'Positive',
    'wasteful':'Negative',
    'stretched':'Negative',
    'inclusive':'Positive',
    'overworked':'Negative',
    'schizophrenic':'Negative',
    'listener':'Positive',
    'slow':'Negative',
    'resilient':'Positive',
    'decent':'Positive',
    'blinkered':'Negative',
    'leave':'Neutral',
    'caring':'Positive',
    'uncaring':'Negative',
    'collaborative':'Positive',
    'professional':'Positive',
    'lacking':'Negative',
    'inflexible':'Negative',
    'inefficient':'Negative'
    
}

sia = SentimentIntensityAnalyzer()

filtered_words = [word.lower() for word in nltk.word_tokenize(' '.join(data['If your Norfolk County Council came to life as a person, what single word would you use to describe it?'].astype(str)))
                  if word.lower() not in stop_words and word.lower() not in punctuation and word.isalnum() 
                  and word.lower() not in exclude_words]

category_counts = {'Positive': [], 'Negative': [], 'Neutral': []}
for word in filtered_words:
    category = word_categories.get(word)
    if category is None:
        sentiment_score = sia.polarity_scores(word)
        if sentiment_score['compound'] >= 0.05:
            category_counts['Positive'].append(word)
        elif sentiment_score['compound'] <= -0.05:
            category_counts['Negative'].append(word)
        else:
            category_counts['Neutral'].append(word)
    else:
        category_counts[category].append(word)

df_positive = pd.DataFrame(nltk.FreqDist(category_counts['Positive']).items(), columns=['Word', 'Count']).sort_values(by='Count', ascending=False).head(50)
df_negative = pd.DataFrame(nltk.FreqDist(category_counts['Negative']).items(), columns=['Word', 'Count']).sort_values(by='Count', ascending=False).head(50)
df_neutral = pd.DataFrame(nltk.FreqDist(category_counts['Neutral']).items(), columns=['Word', 'Count']).sort_values(by='Count', ascending=False).head(50)


df_positive.to_csv('positive_words.csv', index=False)
df_negative.to_csv('negative_words.csv', index=False)
df_neutral.to_csv('neutral_words.csv', index=False)


plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.barh(df_positive['Word'].head(20), df_positive['Count'].head(20), color='green')
plt.xlabel('Count')
plt.title('Top 20 Positive Words')

plt.subplot(2, 2, 2)
plt.barh(df_negative['Word'].head(20), df_negative['Count'].head(20), color='red')
plt.xlabel('Count')
plt.title('Top 20 Negative Words')

plt.subplot(2, 2, 3)
plt.barh(df_neutral['Word'].head(20), df_neutral['Count'].head(20), color='grey')
plt.xlabel('Count')
plt.title('Top 20 Neutral Words')

plt.tight_layout()
plt.show()

