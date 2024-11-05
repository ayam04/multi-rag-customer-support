from typing import List
from utils import send_email_agent, continuous_monitoring
from functions import get_platform_type, search_mongodb_platform, search_vector_database, generate_response, create_vector_database, query_agent
import re

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
            context = search_mongodb_platform(query, "assessments")
        else:
            context = search_vector_database(query)
        
        response = generate_response(context, query)
        return response
        
    except Exception as e:
        print(f"{Colors.RED}Error processing query: {str(e)}{Colors.END}")
        return ""

def update_database():
    try:
        create_vector_database()
        print(f"{Colors.GREEN}Database updated successfully.{Colors.END}")
    except Exception as e:
        print(f"{Colors.RED}Error updating database: {str(e)}{Colors.END}")

def send_emails(recipient_emails: List[str]):
    try:
        for email in recipient_emails:
            send_email_agent(email)
        print(f"{Colors.GREEN}Emails sent successfully.{Colors.END}")
    except Exception as e:
        print(f"{Colors.RED}Failed to send emails: {str(e)}{Colors.END}")

def start_monitoring():
    try:
        print(f"{Colors.BLUE}Starting continuous email monitoring...{Colors.END}")
        continuous_monitoring()
    except Exception as e:
        print(f"{Colors.RED}Error in continuous monitoring: {str(e)}{Colors.END}")

def main_with_agent():
    print(f"{Colors.HEADER}{Colors.BOLD}Welcome to the Customer Support Chatbot!{Colors.END}")
    
    while True:
        query = input(f"{Colors.CYAN}How can I help you? {Colors.END}")
        
        if not query:
            continue
            
        option, processed_query = query_agent(query)
        
        if option == 1:
            response = process_query(processed_query)
            if response:
                print(f"{Colors.GREEN}Response: {Colors.END}{response}")
        elif option == 2:
            print(f"{Colors.YELLOW}Updating database...{Colors.END}")
            update_database()
        elif option == 3:
            if ',' in query:
                emails = re.findall(r'[\w\.-]+@[\w\.-]+', query)
                if emails:
                    send_emails(emails)
                else:
                    emails = input(f"{Colors.CYAN}Enter recipient emails separated by commas: {Colors.END}").split(',')
                    send_emails([email.strip() for email in emails])
            else:
                emails = input(f"{Colors.CYAN}Enter recipient emails separated by commas: {Colors.END}").split(',')
                send_emails([email.strip() for email in emails])
        elif option == 4:
            print(f"{Colors.YELLOW}Starting email monitoring...{Colors.END}")
            start_monitoring()
        elif option == 5:
            print(f"{Colors.BLUE}Goodbye!{Colors.END}")
            break

if __name__ == "__main__":
    main_with_agent()