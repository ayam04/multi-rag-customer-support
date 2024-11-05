import smtplib
import imaplib
import email
import csv
import openai
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from dotenv import load_dotenv
from functions import classify_response, bot_reply
import time

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
openai_client = openai.Client()

SMTP_PORT = 587
CHECK_INTERVAL = 20
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
SENDER_NAME = "HYRGPT"

HTML_TEMPLATE = """
<html>
<body style="font-family: Arial, sans-serif; background-color: #f4f4f4; margin: 0; padding: 20px;">
  <div style="max-width: 600px; background-color: #ffffff; margin: 0 auto; padding: 20px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
    <h1 style="color: #4A90E2; text-align: center;">Streamline Your Hiring Process</h1>
    <p style="font-size: 16px; color: #333333; line-height: 1.6;">
      Hello,
      
      We noticed you might be interested in improving your hiring process. Our solution can help you find the right candidates more efficiently.
    </p>
    <p style="text-align: center;">
      <a href="https://example.com" style="display: inline-block; padding: 12px 24px; background-color: #4A90E2; color: #ffffff; text-decoration: none; border-radius: 4px; font-weight: bold;">
        Learn More
      </a>
    </p>
    <hr style="border: none; border-top: 1px solid #eeeeee; margin: 20px 0;">
    <p style="font-size: 14px; color: #666666; text-align: center;">
      You're receiving this because you've shown interest in hiring solutions. 
      To stop receiving these emails, 
      <a href="https://example.com/unsubscribe?email={{email}}" style="color: #4A90E2; text-decoration: underline;">
        unsubscribe here
      </a>.
    </p>
  </div>
</body>
</html>
"""

TEXT_TEMPLATE = """
Streamline Your Hiring Process

Hello,

We noticed you might be interested in improving your hiring process. Our solution can help you find the right candidates more efficiently.

Learn more at https://example.com.

---

You're receiving this because you've shown interest in hiring solutions. 
To stop receiving these emails, unsubscribe here: https://example.com/unsubscribe?email={{email}}
"""

def get_smtp_server(email_domain):
    domain_to_smtp = {
        'gmail.com': 'smtp.gmail.com',
        'yahoo.com': 'smtp.mail.yahoo.com',
        'outlook.com': 'smtp-mail.outlook.com',
    }
    return domain_to_smtp.get(email_domain, 'smtp.example.com')

def get_imap_server(email_domain):
    domain_to_imap = {
        'gmail.com': 'imap.gmail.com',
        'yahoo.com': 'imap.mail.yahoo.com',
        'outlook.com': 'imap-mail.outlook.com',
    }
    return domain_to_imap.get(email_domain, 'imap.example.com')

def send_email_agent(recipient_email):
    email_domain = recipient_email.split('@')[1]
    smtp_server = get_smtp_server(email_domain)
    
    msg = MIMEMultipart('alternative')
    msg['From'] = f"{SENDER_NAME} <{EMAIL_USER}>"
    msg['To'] = recipient_email
    msg['Subject'] = "Improve Your Hiring Process"
    msg['Reply-To'] = EMAIL_USER
    msg['List-Unsubscribe'] = f"<https://example.com/unsubscribe?email={recipient_email}>"

    text_content = TEXT_TEMPLATE.format(email=recipient_email)
    html_content = HTML_TEMPLATE.format(email=recipient_email)

    msg.attach(MIMEText(text_content, 'plain'))
    msg.attach(MIMEText(html_content, 'html'))

    try:
        with smtplib.SMTP(smtp_server, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_USER, EMAIL_PASS)
            server.send_message(msg)
        print(f"Email sent to {recipient_email}")
    except Exception as e:
        print(f"Error sending email to {recipient_email}: {str(e)}")

def reply_agent():
    email_domain = EMAIL_USER.split('@')[1]
    imap_server = get_imap_server(email_domain)
    try:
        mail = imaplib.IMAP4_SSL(imap_server)
        mail.login(EMAIL_USER, EMAIL_PASS)
        mail.select('inbox')

        _, data = mail.search(None, '(UNSEEN)')
        email_ids = data[0].split()

        for e_id in email_ids:
            _, email_data = mail.fetch(e_id, '(RFC822)')
            raw_email = email_data[0][1].decode('utf-8')
            msg = email.message_from_string(raw_email)
            sender = msg['from']
            body = get_body(msg)
            
            response_category = classify_response(body)
            print(f"Received email from {sender} classified as {response_category}")
            save_to_csv(sender, body, response_category)
            send_reply(sender, response_category)

        mail.logout()
    except Exception as e:
        print(f"Error in reply_agent: {str(e)}")

def get_body(msg):
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == 'text/plain':
                return part.get_payload(decode=True).decode('utf-8')
    else:
        return msg.get_payload(decode=True).decode('utf-8')

def save_to_csv(sender, body, category):
    file_exists = os.path.isfile('responses.csv')
    with open('responses.csv', 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Sender', 'Response', 'Category']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({'Sender': sender, 'Response': body, 'Category': category})
    print(f"Saved response from {sender} to CSV.")

def send_reply(recipient_email, category):
    try:
        data = bot_reply(category)
        email_domain = EMAIL_USER.split('@')[1]
        smtp_server = get_smtp_server(email_domain)
        
        msg = MIMEMultipart('alternative')
        msg['From'] = f"{SENDER_NAME} <{EMAIL_USER}>"
        msg['To'] = recipient_email
        msg['Subject'] = data['subject']
        msg['Reply-To'] = EMAIL_USER

        text_content = data['body']
        html_content = f"<html><body>{data['body']}</body></html>"

        msg.attach(MIMEText(text_content, 'plain'))
        msg.attach(MIMEText(html_content, 'html'))

        with smtplib.SMTP(smtp_server, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_USER, EMAIL_PASS)
            server.send_message(msg)
        
        print(f"Reply sent to {recipient_email} for {category} response")
    except Exception as e:
        print(f"Error in send_reply: {str(e)}")

def continuous_monitoring():
    print("Starting continuous email monitoring...")
    while True:
        try:
            reply_agent()
            interval = 10
            print(f"Checked for new emails. Waiting for {interval} seconds before next check.")
            time.sleep(interval)
        except KeyboardInterrupt:
            print("Monitoring stopped by user.")
            break
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            print(f"Restarting monitoring in {CHECK_INTERVAL} seconds...")
            time.sleep(CHECK_INTERVAL)