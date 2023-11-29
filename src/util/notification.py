import requests
import json
import os
from dotenv import load_dotenv

# .env ファイルから環境変数をロード
load_dotenv()

class Notification:
    def send_slack_notification(self, message, channel="#general", username="Notification"):
        """
        Slackに通知を送るメソッド

        :param webhook_url: Slack Incoming WebhookのURL
        :param message: 送信するメッセージ
        :param channel: 送信先のチャンネル(デフォルトは#general)
        :param username: 送信者のユーザー名(デフォルトはNotification Bot)
        """
        payload = {
            "channel": channel,
            "username": username,
            "text": message,
        }
        webhook_url = os.environ['SLACK_WEBHOOK_URL']


        try:
            response = requests.post(webhook_url, data=json.dumps(payload), headers={'Content-Type': 'application/json'})
            response.raise_for_status()
            print("Slack notification sent successfully!")
        except requests.exceptions.RequestException as err:
            print(f"Failed to send Slack notification: {err}")

    def send_line_notification(self, message):
        url = "https://notify-api.line.me/api/notify"
        line_token = os.environ['LINE_TOKEN']

        headers = {"Authorization": f"Bearer {line_token}",}
        payload = {"message": message}

        try:
            response = requests.post(url, data=payload, headers=headers)
            response.raise_for_status()
            print("Line notification sent successfully!")
        except requests.exceptions.RequestException as err:
            print(f"Failed to send Line notification: {err}")

# notification.send_line_notification(message="hello")
# notification.send_slack_notification(message="hello", channel="vit-team")