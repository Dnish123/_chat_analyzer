import emoji
import pandas as pd
from collections import Counter
from wordcloud import WordCloud
import re

def fetch_stats(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
        #1. fetch messages
    num_messages = df.shape[0]
        #2. number of words
    words =[]
    for message in df['message']:
        words.extend(message.split())

     # fetch number of media messages

    num_media_msg = df[df['message'] == '<Media omitted>\n'].shape[0]
    # fetch emojis
    emojis = []
    for message in df['message']:
        for char in message:
            if char in emoji.EMOJI_DATA:
                emojis.append(char)

    total_emojis = len(emojis)
    top_emojis = None
    if emojis:
        top_emojis = Counter(emojis).most_common(1)[0]  # frequency of each emoji

    return num_messages, len(words),num_media_msg,total_emojis,top_emojis
def most_busy_users(df):
    x = df['user'].value_counts().head()
    df = (df['user'].value_counts(normalize=True) * 100).round(2).reset_index().rename(
        columns={'index': 'name', 'user': 'percent'}
    )

    return x,df
def create_wordcloud(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    with open('stop_hinglish.txt', 'r', encoding='utf-8') as f:
        stop_words = set(f.read().split())

    # remove group notifications and media/deleted/edited messages
    temp = df[df['user'] != 'group_notification']
    temp = temp[~temp['message'].str.contains('Media omitted|deleted|edited', case=False, na=False)]

    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return " ".join(y)

    temp['message'] = temp['message'].apply(remove_stop_words)

    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
    df_wc = wc.generate(temp['message'].str.cat(sep=" "))
    return df_wc


def most_common_words(selected_user, df):
    with open('stop_hinglish.txt', 'r', encoding='utf-8') as f:
        stop_words = set(f.read().split())

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # remove group notifications
    temp = df[df['user'] != 'group_notification']

    # remove system messages like deleted / media / edited
    temp = temp[~temp['message'].str.contains('Media omitted|deleted|edited', case=False, na=False)]

    # ✅ get all participant names in lowercase to exclude them
    usernames = {u.lower() for u in df['user'].unique()}

    words = []
    for message in temp['message']:
        # remove mentions (@username) and non-alphabetic stuff
        message = re.sub(r'@\w+', '', message)      # remove @mentions
        message = re.sub(r'<.*?>', '', message)     # remove things in < >
        message = re.sub(r'[^a-zA-Z\s]', '', message)  # keep only letters and spaces

        for word in message.lower().split():
            if word not in stop_words and word not in usernames and len(word) > 1:  # ✅ exclude usernames
                words.append(word)

    most_common_df = pd.DataFrame(Counter(words).most_common(25))
    return most_common_df
def monthly_timeline(selected_user,df):


    if selected_user != 'Overall':
        df =df[df['user'] == selected_user]
    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))
    timeline['time'] = time
    return timeline

def daily_timeline(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    daily_timeline = df.groupby('only_date').count()['message'].reset_index()

    return daily_timeline

def week_activity_map(selected_user, df):

     if selected_user != 'Overall':
         df =df[df['user']==selected_user]

     return df['day_name'].value_counts()

def month_activity_map(selected_user, df):

     if selected_user != 'Overall':
         df =df[df['user']==selected_user]

     return df['month'].value_counts()

def activity_heatmap(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)
    return user_heatmap