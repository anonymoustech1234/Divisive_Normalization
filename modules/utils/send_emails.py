import smtplib
from email.mime.text import MIMEText

SENDER = "YOUR EMAIL ADDRESS FOR SENDING THE RESULT"
RECIPENTS = ["EMAIL ADDRESSES FOR RECEIVING THE RESULTS"]
PASSWORD = "SETTED APP TOKEN PASSWORD FOR SENDER EMAIL"

def send_emails(subject, body, recipents=RECIPENTS, sender=SENDER, password=PASSWORD):
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = ', '.join(recipents)
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp_server:
        smtp_server.login(sender, password)
        smtp_server.sendmail(sender, recipents, msg.as_string())

def format_result(train_loss, train_acc, val_loss, val_acc, runtime) -> str:

    run_time_in_hour = round(runtime / 3600, 2)
    body = ''

    body += ("Best Validation Accuracy: " + str(round(max(val_acc), 2)) + "\n")
    body += ("Experiment Run Time(hrs): " + str(run_time_in_hour))
    body += "\n\n\n\n\n"

    body += ("Train Loss" + str(train_loss) + "\n\n")
    body += ("Train Accuracy" + str(train_acc) + "\n\n")
    body += ("Validation Loss" + str(val_loss) + "\n\n")
    body += ("Validation Accuracy" + str(val_acc) + "\n\n")

    return body
