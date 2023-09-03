import webbrowser

def open_git_hub():
    """
    Opens the GitHub repository for the DLTA-AI project in the default web browser.

    Parameters:
    None

    Returns:
    None
    """

    # Open the GitHub repository in the default web browser
    webbrowser.open('https://github.com/0ssamaak0/DLTA-AI')  

def open_issue():
    """
    Opens the GitHub issues page for the DLTA-AI project in the default web browser.

    Parameters:
    None

    Returns:
    None
    """

    # Open the GitHub issues page in the default web browser
    webbrowser.open('https://github.com/0ssamaak0/DLTA-AI/issues')

def open_license():
    """
    Opens the license file for the DLTA-AI project in the default web browser.

    Parameters:
    None

    Returns:
    None
    """

    # Open the license file in the default web browser
    webbrowser.open('https://github.com/0ssamaak0/DLTA-AI/blob/master/LICENSE')

def open_guide():
    """
    Opens the guide for the DLTA-AI project in the default web browser.

    Parameters:
    None

    Returns:
    None
    """

    # Open the guide in the default web browser
    webbrowser.open('https://0ssamaak0.github.io/DLTA-AI/')

def open_release(link = None):
    """
    Opens the release page for the DLTA-AI project in the default web browser.

    Parameters:
    link (str): The link to the release page. If None, the default link will be used.

    Returns:
    None
    """
    # Import necessary modules
    import webbrowser

    # If no link was provided, use the default link
    if link is None:
        link = 'https://github.com/0ssamaak0/DLTA-AI/releases'
    else:
        link = "https://github.com/" + link

    # Open the release page in the default web browser
    webbrowser.open(link)
