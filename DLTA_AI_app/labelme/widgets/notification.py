import os

def PopUp(text):
    """
    Sends a desktop notification with the given text.

    Args:
        text (str): The text to display in the notification.

    Returns:
        None
    """
    try:
        from notifypy import Notify
        # Create a Notify object with the default title
        notification = Notify(default_notification_title="DLTA-AI")

        # Set the message of the notification to the given text
        notification.message = text

        # Set the notification icon
        print(os.getcwd())
        notification.icon = "labelme/icons/icon.ico"

        # Send the notification asynchronously
        notification.send(block=False)
    except Exception as e:
        print(e)
        print("please install notifypy to get desktop notifications")
