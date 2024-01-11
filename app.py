import requests
from datetime import datetime
from dotenv import load_dotenv
import psycopg2
import os

load_dotenv()
access_key = os.getenv('discord_authorization_key')
def parse_timestamp(timestamp_str):
    try:
        return datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%S.%f%z')
    except ValueError:
        return datetime.fromisoformat(timestamp_str)

def get_server_name(guild_id, headers):
    r_guild = requests.get(f'https://discord.com/api/v9/guilds/{guild_id}', headers=headers)
    if r_guild.status_code == 200:
        guild_data = r_guild.json()
        return guild_data.get('name', 'Unknown Server')
    else:
        print(f"Error retrieving guild information for ID {guild_id}. Status code: {r_guild.status_code}")
        return 'Unknown Server'

def insert_message(server, channel, author, original_name, message, time_stamp,data_source):
    try:
        connection = psycopg2.connect(os.getenv('DATABASE_URL'))
        cursor = connection.cursor()

        sql = """
        INSERT INTO data (server, channel, author, original_name, message, time_stamp,data_source)
        VALUES (%s, %s, %s, %s, %s, %s,%s)
        """
        cursor.execute(sql, (server, channel, author, original_name, message, time_stamp,data_source))

        connection.commit()
    except (Exception, psycopg2.Error) as error:
        print("Error while connecting to PostgreSQL:", error)
    finally:
        if connection:
            cursor.close()
            connection.close()

def retrieve_messages(group_channels):
    headers = {
        'Authorization': access_key
    }

    for group_id, channel_ids in group_channels:
        # Optionally, you can retrieve information about the group (category)
        r_group = requests.get(f'https://discord.com/api/v9/guilds/{group_id}/channels', headers=headers)
        if r_group.status_code == 200:
            json_data_group = r_group.json()
            print(json_data_group)
            server_name = get_server_name(group_id, headers)
            print(f'Channels in Group {group_id} (Server: {server_name}):')
            for channel in json_data_group:
                print(f"Channel ID: {channel['id']}, Channel Name: {channel['name']}")
            print(f'\n--- End of Group {server_name} ---\n')
        else:
            print(f"Error retrieving channels in group {group_id}. Status code: {r_group.status_code}")
            continue

        # Retrieve messages from each channel in the group
        for channel_id in channel_ids:
            # Initialize variables for pagination
            messages = []
            last_message_id = None

            url = f'https://discord.com/api/v9/channels/{channel_id}/messages'
            params = {'limit': 50}  # Adjust the limit as needed (max is 100)

            r_channel = requests.get(url, headers=headers, params=params)
            if r_channel.status_code == 200:
                json_data_channel = r_channel.json()
                if not json_data_channel:
                    break  # No messages in the channel

                # Sort messages by timestamp in descending order
                messages.extend(sorted(json_data_channel, key=lambda x: parse_timestamp(x['timestamp']), reverse=True))
                last_message_id = messages[-1]['id']

                # Get the server name
                server_name = get_server_name(group_id, headers)
                # Get the channel name
                channel_name = next((channel['name'] for channel in json_data_group if channel['id'] == channel_id), None)

                print(f'Messages from Channel {channel_name} in Server {server_name}:')
                for message in messages:
                    timestamp = parse_timestamp(message['timestamp'])
                    author_name = message['author']['username']
                    original_name = message.get('author', {}).get('member', {}).get('nick', author_name)
                    content = message['content'] if message['content'].strip() != "" else "<Empty Message>"
                    data_source = 'discord'
                    print(f"Author: {author_name}, Original Name: {original_name}, Server: {server_name}, "
                          f"Channel: {channel_name}, Message: {content}, Timestamp: {timestamp}")

                    insert_message(server_name, channel_name, author_name, original_name, content, timestamp,data_source)

                print(f'\n--- End of Channel {channel_name} in Group {server_name} ---\n')
            else:
                print(f"Error retrieving messages from channel {channel_id}. Status code: {r_channel.status_code}")


group_channels = [
    ('905908516894670928', ['1014574494502891551', '1014989330177077370']),
    ('884204406189490176', ['894619517441957908', '895350107137011723'])
]

retrieve_messages(group_channels)