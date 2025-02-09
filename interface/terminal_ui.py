# TODO: Implement this module
# interface/terminal_ui.py
from query_engine.query_handler import process_query

def main():
    print("PUMA - Project Understanding & Modification Accelerator")
    print("Type 'exit()' to quit.")

    while True:
        user_input = input("\n[Enter your query]: ")
        if user_input.lower() == "exit()":
            break

        response = process_query(user_input)
        print("\n[Response]:")
        print(response)

if __name__ == "__main__":
    main()