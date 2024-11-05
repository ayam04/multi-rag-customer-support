from typing import List
from utils import send_email_agent, continuous_monitoring
from functions import get_platform_type, search_mongodb_platform, search_vector_database, generate_response, create_vector_database

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def process_query(query: str) -> str:
    try:
        if not query:
            print(f"{Colors.RED}Error: Query is required.{Colors.END}")
            return ""
        
        platform = get_platform_type(query)
        
        if platform == "mongodb":
            context = search_mongodb_platform(query, "assessment")
        else:
            context = search_vector_database(query)
        
        response = generate_response(context, query)
        return response
        
    except Exception as e:
        print(f"{Colors.RED}Error processing query: {str(e)}{Colors.END}")
        return ""

def update_database():
    """
    Update the vector database.
    """
    try:
        create_vector_database()
        print(f"{Colors.GREEN}Database updated successfully.{Colors.END}")
    except Exception as e:
        print(f"{Colors.RED}Error updating database: {str(e)}{Colors.END}")

def send_emails(recipient_emails: List[str]):
    """
    Send emails to the specified list of recipients.
    """
    try:
        for email in recipient_emails:
            send_email_agent(email)
        print(f"{Colors.GREEN}Emails sent successfully.{Colors.END}")
    except Exception as e:
        print(f"{Colors.RED}Failed to send emails: {str(e)}{Colors.END}")

def start_monitoring():
    """
    Start continuous email monitoring.
    """
    try:
        print(f"{Colors.BLUE}Starting continuous email monitoring...{Colors.END}")
        continuous_monitoring()
    except Exception as e:
        print(f"{Colors.RED}Error in continuous monitoring: {str(e)}{Colors.END}")

def main():
    """
    Main interactive loop for the terminal chatbot.
    """
    print(f"{Colors.HEADER}{Colors.BOLD}Welcome to the Customer Support Chatbot!{Colors.END}")
    while True:
        print(f"\n{Colors.CYAN}Options:{Colors.END}")
        print(f"{Colors.YELLOW}1. Query the chatbot{Colors.END}")
        print(f"{Colors.YELLOW}2. Update the database{Colors.END}")
        print(f"{Colors.YELLOW}3. Send emails{Colors.END}")
        print(f"{Colors.YELLOW}4. Start continuous email monitoring{Colors.END}")
        print(f"{Colors.YELLOW}5. Exit{Colors.END}")
        
        choice = input(f"{Colors.CYAN}Enter your choice (1-5): {Colors.END}")

        if choice == '1':
            query = input(f"{Colors.CYAN}Enter your query: {Colors.END}")
            response = process_query(query)
            if response:
                print(f"{Colors.GREEN}Response: {Colors.END}{response}")
        elif choice == '2':
            update_database()
        elif choice == '3':
            emails = input(f"{Colors.CYAN}Enter recipient emails separated by commas: {Colors.END}").split(',')
            send_emails([email.strip() for email in emails])
        elif choice == '4':
            start_monitoring()
        elif choice == '5':
            print(f"{Colors.BLUE}Exiting...{Colors.END}")
            break
        else:
            print(f"{Colors.RED}Invalid choice. Please enter a number between 1 and 5.{Colors.END}")

if __name__ == "__main__":
    main()