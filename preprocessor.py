import re
import pandas as pd
from datetime import datetime

def convert_chat_format(data: str) -> str:
    """
    Converts WhatsApp chat timestamps from DD/MM/YYYY, hh:mm am/pm
    to M/D/YY, HH:MM (24-hour format, WhatsApp style).
    Leaves already converted files unchanged.
    """
    pattern = r"(\d{1,2}/\d{1,2}/\d{4}), (\d{1,2}:\d{2})\s?(am|pm)?"

    def replacer(match):
        date_str = match.group(1)
        time_str = match.group(2)
        ampm = match.group(3)

        if ampm:  # Handle 12-hour clock with am/pm
            dt = datetime.strptime(f"{date_str}, {time_str} {ampm}", "%d/%m/%Y, %I:%M %p")
        else:  # Handle already 24-hour clock
            dt = datetime.strptime(f"{date_str}, {time_str}", "%d/%m/%Y, %H:%M")

        # Format into WhatsApp 24hr style: M/D/YY, HH:MM - (no leading zeros)
        formatted = dt.strftime("%m/%d/%y, %H:%M - ")
        return formatted.lstrip("0").replace("/0", "/")

    return re.sub(pattern, replacer, data, flags=re.IGNORECASE)


def preprocess(data):
    # Step 1: Convert timestamps first
    data = convert_chat_format(data)

    # Step 2: Continue with your existing preprocessing logic
    pattern = '\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s-\s'

    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)

    df = pd.DataFrame({'user_message': messages, 'message_date': dates})
    df['message_date'] = pd.to_datetime(df['message_date'], format='%m/%d/%y, %H:%M - ')
    df.rename(columns={'message_date': 'date'}, inplace=True)

    users = []
    messages = []
    for message in df['user_message']:
        entry = re.split(r'^([^:]+):\s', message)
        if entry[1:]:  # user name
            users.append(entry[1])
            messages.append(entry[2])
        else:
            users.append('group_notification')
            messages.append(entry[0])

    df['user'] = users
    df['message'] = messages
    df.drop(columns=['user_message'], inplace=True)

    df['year'] = df['date'].dt.year
    df['only_date'] = df['date'].dt.date
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute

    period = []
    for hour in df[['day_name', 'hour']]['hour']:
        if hour == 23:
            period.append(str(hour) + "-" + str('00'))
        elif hour == 0:
            period.append(str('00') + "-" + str(hour + 1))
        else:
            period.append(str(hour) + "-" + str(hour + 1))

    df['period'] = period
    return df
